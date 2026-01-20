# Cluster 86

def split_coco(data_root, out_dir, percent, fold):
    """Split COCO data for Semi-supervised object detection.

    Args:
        data_root (str): The data root of coco dataset.
        out_dir (str): The output directory of coco semi-supervised
            annotations.
        percent (float): The percentage of labeled data in the training set.
        fold (int): The fold of dataset and set as random seed for data split.
    """

    def save_anns(name, images, annotations):
        sub_anns = dict()
        sub_anns['images'] = images
        sub_anns['annotations'] = annotations
        sub_anns['licenses'] = anns['licenses']
        sub_anns['categories'] = anns['categories']
        sub_anns['info'] = anns['info']
        mmcv.mkdir_or_exist(out_dir)
        mmcv.dump(sub_anns, f'{out_dir}/{name}.json')
    np.random.seed(fold)
    ann_file = osp.join(data_root, 'annotations/instances_train2017.json')
    anns = mmcv.load(ann_file)
    image_list = anns['images']
    labeled_total = int(percent / 100.0 * len(image_list))
    labeled_inds = set(np.random.choice(range(len(image_list)), size=labeled_total))
    labeled_ids, labeled_images, unlabeled_images = ([], [], [])
    for i in range(len(image_list)):
        if i in labeled_inds:
            labeled_images.append(image_list[i])
            labeled_ids.append(image_list[i]['id'])
        else:
            unlabeled_images.append(image_list[i])
    labeled_ids = set(labeled_ids)
    labeled_annotations, unlabeled_annotations = ([], [])
    for ann in anns['annotations']:
        if ann['image_id'] in labeled_ids:
            labeled_annotations.append(ann)
        else:
            unlabeled_annotations.append(ann)
    labeled_name = f'instances_train2017.{fold}@{percent}'
    unlabeled_name = f'instances_train2017.{fold}@{percent}-unlabeled'
    save_anns(labeled_name, labeled_images, labeled_annotations)
    save_anns(unlabeled_name, unlabeled_images, unlabeled_annotations)

def save_anns(name, images, annotations):
    sub_anns = dict()
    sub_anns['images'] = images
    sub_anns['annotations'] = annotations
    sub_anns['licenses'] = anns['licenses']
    sub_anns['categories'] = anns['categories']
    sub_anns['info'] = anns['info']
    mmcv.mkdir_or_exist(out_dir)
    mmcv.dump(sub_anns, f'{out_dir}/{name}.json')

def multi_wrapper(args):
    return split_coco(*args)

