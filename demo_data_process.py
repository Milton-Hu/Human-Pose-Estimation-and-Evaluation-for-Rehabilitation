# 绘图相关包
from PyQt5.Qt import *
from pyqtgraph import PlotWidget
from PyQt5 import QtCore
import numpy as np
import pyqtgraph as pq

# 绘图相关包
import cv2
import math
import torch
import pyrealsense2 as rs

#人体姿态估计相关包
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints, BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width

KPTS_ANGLE_PAIRS = [[2,3,4], [5,6,7], [8,9,10], [14,15,16], [9,10,12], [15,16,18]]

# Q为这一轮的心里的预估误差
Q = 0.00001
# R为下一轮的测量误差
R = 0.1
# Accumulated_Error为上一轮的估计误差，具体呈现为所有误差的累计
Accumulated_Error = np.ones(6, np.float32)
# 初始旧值
kalman_kpt_old = np.zeros(6, np.float32)

SCOPE = 50


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad

def pose_estimate(net, img, depth_img, height_size, cpu, track, smooth):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts  # num_kpts = 18
    previous_poses = []
    keypoint_angel = np.zeros(6,np.float32)   # 重要关节角度值
    delay = 1
    camera_px = 331.232
    camera_py = 252.661
    camera_fx = 611.462
    camera_fy = 610.139

    orig_img = img.copy()
    heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs) # pose_entries是不同人体的pose
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
    current_poses = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
        pose = Pose(pose_keypoints, pose_entries[n][24]) #TODO:这里原来是18，不知道是用于什么
        current_poses.append(pose)  # 存储不同人体的pose信息
    
    if track:
        track_poses(previous_poses, current_poses, smooth=smooth)
        previous_poses = current_poses
    for pose in current_poses:
        pose.draw(img)
        for idx, pair in enumerate(KPTS_ANGLE_PAIRS):  # 对肘、膝关节进行关节角度测量
            joint_forward = (pose.keypoints[pair[0]][1], pose.keypoints[pair[0]][0])
            joint = (pose.keypoints[pair[1]][1], pose.keypoints[pair[1]][0])
            joint_backward = (pose.keypoints[pair[2]][1], pose.keypoints[pair[2]][0])
            if joint[0]*joint[1] > -1:
                if joint_forward[0]*joint_forward[1] > 1 and joint_backward[0]*joint_backward[1] > 1:
                    joint_forward_depth = depth_img[joint_forward[0], joint_forward[1]] * 0.001
                    joint_depth = depth_img[joint[0], joint[1]] * 0.001
                    joint_backward_depth = depth_img[joint_backward[0], joint_backward[1]] * 0.001
                    if joint_backward_depth * joint_depth * joint_forward_depth > 0:
                        joint_forward_location = ( - (joint_forward[0] - camera_px) * joint_forward_depth / camera_fx, - (joint_forward[1] - camera_py) * joint_forward_depth / camera_fy, joint_forward_depth)
                        joint_location = ( - (joint[0] - camera_px) * joint_depth / camera_fx, - (joint[1] - camera_py) * joint_depth / camera_fy, joint_depth)
                        joint_backward_location = ( - (joint_backward[0] - camera_px) * joint_backward_depth / camera_fx, - (joint_backward[1] - camera_py) * joint_backward_depth / camera_fy, joint_backward_depth)
                        joint_angle = cal_joint_angle(joint_forward_location, joint_location, joint_backward_location)

                        keypoint_angel[idx] = joint_angle  #获得关节角度

                        # cv2.rectangle(img, (joint[1], joint[0]), (joint[1]+45, joint[0]-12), (255,255,255), thickness=-1)
                        # cv2.putText(img, '%.1f'%joint_angle+'', (joint[1], joint[0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), thickness=1)
                    # else:
                        # cv2.rectangle(img, (joint[1], joint[0]), (joint[1]+45, joint[0]-12), (255,255,255), thickness=-1)
                        # cv2.putText(img, 'Null', (joint[1], joint[0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), thickness=1)
            
    for pose in current_poses:
        cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                        (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
        if track:
            cv2.putText(img, 'human pose: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
    return keypoint_angel, img

def cal_joint_angle(point1, point2, point3):
    # 计算三点的角度，以点2为顶点
    a=math.sqrt((point2[0]-point3[0])*(point2[0]-point3[0])+(point2[1]-point3[1])*(point2[1] - point3[1])+(point2[2]-point3[2])*(point2[2] - point3[2]))
    b=math.sqrt((point1[0]-point3[0])*(point1[0]-point3[0])+(point1[1]-point3[1])*(point1[1] - point3[1])+(point1[2]-point3[2])*(point1[2] - point3[2]))
    c=math.sqrt((point1[0]-point2[0])*(point1[0]-point2[0])+(point1[1]-point2[1])*(point1[1] - point2[1])+(point1[2]-point2[2])*(point1[2] - point2[2]))
    
    B=math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c)))
    return B

def kalman(keypoint_angle):
    global kalman_kpt_old
    global Accumulated_Error
    kalman_kpt = np.zeros(6, np.float32)

    for i in range(len(keypoint_angle)):
        #处理异常点
        if(keypoint_angle[i] == 0 and kalman_kpt_old[i]-keypoint_angle[i]>40):
            keypoint_angle[i] = kalman_kpt_old[i]

        # 新的值相比旧的值差太大时进行跟踪
        if (abs(keypoint_angle[i]-kalman_kpt_old[i])/SCOPE > 0.25):
            Old_Input = keypoint_angle[i]*0.382 + kalman_kpt_old[i]*0.618
        else:
            Old_Input = kalman_kpt_old[i]

        # 上一轮的 总误差=累计误差^2+预估误差^2
        Old_Error_All = (Accumulated_Error[i]**2 + Q**2)**(1/2)

        # R为这一轮的预估误差
        # H为利用均方差计算出来的双方的相信度
        H = Old_Error_All**2/(Old_Error_All**2 + R**2)

        # 旧值 + 1.00001/(1.00001+0.1) * (新值-旧值)
        kalman_kpt[i] = Old_Input + H * (keypoint_angle[i] - Old_Input)

        # 计算新的累计误差
        Accumulated_Error[i] = ((1 - H)*Old_Error_All**2)**(1/2)
        # 新值变为旧值
        kalman_kpt_old[i] = kalman_kpt[i]
    return kalman_kpt

class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.win = pq.GraphicsLayoutWidget(show=True)
        self.win.setWindowTitle('人体关节角度值')

        self.p1 = self.win.addPlot(title="右肘")
        self.p1.setRange(yRange = [0,180])
        self.p2 = self.win.addPlot(title="左肘")
        self.p2.setRange(yRange = [0,180])
        self.win.nextRow()
        self.p3 = self.win.addPlot(title="右膝")
        self.p3.setRange(yRange = [0,180])
        self.p4 = self.win.addPlot(title="左膝")
        self.p4.setRange(yRange = [0,180])
        self.win.nextRow()
        self.p5 = self.win.addPlot(title="右脚")
        self.p5.setRange(yRange = [0,180])
        self.p6 = self.win.addPlot(title="左脚")
        self.p6.setRange(yRange = [0,180])

        self.keypoints_angel = np.zeros([6,1000],np.float32)
        # self.keypoint_angel_filtered = np.zeros(1000,np.float32)
        
        self.curve1 = self.p1.plot(self.keypoints_angel[0,:], name="mode1")
        # self.curve1_1 = self.p1.plot(self.keypoint_angel_filtered, name="mode1", pen='w')
        self.curve2 = self.p2.plot(self.keypoints_angel[1,:], name="mode1")
        self.curve3 = self.p3.plot(self.keypoints_angel[2,:], name="mode1")
        self.curve4 = self.p4.plot(self.keypoints_angel[3,:], name="mode1")
        self.curve5 = self.p5.plot(self.keypoints_angel[4,:], name="mode1")
        self.curve6 = self.p6.plot(self.keypoints_angel[5,:], name="mode1")

        # 设定定时器
        self.timer = pq.QtCore.QTimer()
        self.timer_drow = pq.QtCore.QTimer()
        # 定时器信号绑定 update_data 函数
        self.timer.timeout.connect(self.update_data)
        self.timer_drow.timeout.connect(self.display)
        # 定时器间隔100ms
        self.timer.start(50)
        self.timer_drow.start(100)

        # 相机初始化
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)
        self.pipeline.start(self.config)

        # 网络加载
        self.net = PoseEstimationWithMobileNet()
        self.checkpoint_path = 'checkpoint_iter_33000.pth'
        self.checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        load_state(self.net, self.checkpoint)


    # 数据左移
    def update_data(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        self.keypoints_angel[:,:-1] = self.keypoints_angel[:,1:]
        # self.keypoint_angel_filtered[:-1] = self.keypoint_angel_filtered[1:]

        keypoints_angel, self.img = pose_estimate(self.net, color_image, depth_image, height_size=256, cpu=0, track=1, smooth=1)
        
        # 对每个关键角度做卡尔曼滤波
        keypoints_angel_filtered = kalman(keypoints_angel)
        
        for i in range(len(keypoints_angel_filtered)):
            self.keypoints_angel[i,-1] = keypoints_angel_filtered[i]
            # self.keypoints_angel[i,-1] = keypoints_angel[i]

        

        # 数据填充到绘制曲线中
        self.curve1.setData(self.keypoints_angel[0,:])
        # self.curve1_1.setData(self.keypoint_angel_filtered)
        self.curve2.setData(self.keypoints_angel[1,:])
        self.curve3.setData(self.keypoints_angel[2,:])
        self.curve4.setData(self.keypoints_angel[3,:])
        self.curve5.setData(self.keypoints_angel[4,:])
        self.curve6.setData(self.keypoints_angel[5,:])

    def display(self):
        cv2.waitKey(1)
        cv2.imshow('Rehabilitation Action Assessment', self.img)
        # np.savetxt('data.csv', self.keypoints_angel, delimiter = ',')
    
    def __del__(self):
        cv2.destroyAllWindows()
        self.pipeline.stop()


if __name__ == '__main__':
    import sys
    # PyQt5 程序固定写法
    app = QApplication(sys.argv)

    # 将绑定了绘图控件的窗口实例化并展示
    window = Window()
    # window.show()  #TODO: 这个会多出一个空白窗口，注释掉就不会出现了

    # PyQt5 程序固定写法
    sys.exit(app.exec())