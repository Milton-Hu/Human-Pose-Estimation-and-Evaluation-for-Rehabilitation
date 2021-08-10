import argparse
import json
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=str, required=True, help='path to json with keypoints val labels')
    parser.add_argument('--output-name', type=str, default='val_subset.json',
                        help='name of output file with subset of val labels')
    parser.add_argument('--num-images', type=int, default=50, help='number of images in subset')
    args = parser.parse_args()

    with open(args.labels, 'r') as f:
        data = json.load(f)

    random.seed(5)
    total_val_images = 5000
    idxs = list(range(total_val_images))
    random.shuffle(idxs)

    images_by_id = {}
    for idx in idxs[:args.num_images]:
        images_by_id[data['images'][idx]['id']] = data['images'][idx]

    annotations_by_image_id = {}
    for annotation in data['annotations']:
        if annotation['image_id'] in images_by_id:
            if not annotation['image_id'] in annotations_by_image_id:
                annotations_by_image_id[annotation['image_id']] = []
            annotations_by_image_id[annotation['image_id']].append(annotation)

    subset = {
        'info': data['info'],
        'licenses': data['licenses'],
        'images': [],
        'annotations': [],
        'categories': data['categories']
    }
    for image_id, image in images_by_id.items():
        subset['images'].append(image)
        if image_id in annotations_by_image_id:  # image has at least 1 annotation
            subset['annotations'].extend(annotations_by_image_id[image_id])

    with open(args.output_name, 'w') as f:
        json.dump(subset, f, indent=4)
    
    # 修改输出json文件的格式，不然val阶段读取会出现索引需要int型而实际是str的问题
    annFile_foot = args.output_name
    annFile_foot_modified = 'val_subset_modified.json'

    with open(annFile_foot) as f:
        data = json.loads(f.read())
        
        #add additional brackets to categories
        data['categories']=[data['categories']] 
        
        #add additional brakets to annotations
        for i in range(len(data['annotations'])):
            if type(data['annotations'][i]['segmentation'][0])!=list:
                data['annotations'][i]['segmentation'] = [data['annotations'][i]['segmentation']] #additional brackets
        
        #export
        with open(annFile_foot_modified, 'w+') as ff:
            ff.write(json.dumps(data))

