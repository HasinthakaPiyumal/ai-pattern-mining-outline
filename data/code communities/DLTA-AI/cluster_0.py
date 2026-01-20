# Cluster 0

def main():
    version = get_version()
    if sys.argv[1] == 'release':
        if not distutils.spawn.find_executable('twine'):
            print('Please install twine:\n\n\tpip install twine\n', file=sys.stderr)
            sys.exit(1)
        commands = ['python tests/docs_tests/man_tests/test_labelme_1.py', 'git tag v{:s}'.format(version), 'git push origin master --tag', 'python setup.py sdist', 'twine upload dist/labelme-{:s}.tar.gz'.format(version)]
        for cmd in commands:
            print('+ {:s}'.format(cmd))
            subprocess.check_call(shlex.split(cmd))
        sys.exit(0)
    setup(name='labelme', version=version, packages=find_packages(exclude=['github2pypi']), description='Image Polygonal Annotation with Python', long_description=get_long_description(), long_description_content_type='text/markdown', author='Kentaro Wada', author_email='www.kentaro.wada@gmail.com', url='https://github.com/wkentaro/labelme', install_requires=get_install_requires(), license='GPLv3', keywords='Image Annotation, Machine Learning', classifiers=['Development Status :: 5 - Production/Stable', 'Intended Audience :: Developers', 'Natural Language :: English', 'Programming Language :: Python', 'Programming Language :: Python :: 2.7', 'Programming Language :: Python :: 3.5', 'Programming Language :: Python :: 3.6', 'Programming Language :: Python :: 3.7', 'Programming Language :: Python :: Implementation :: CPython', 'Programming Language :: Python :: Implementation :: PyPy'], package_data={'labelme': ['icons/*', 'config/*.yaml']}, entry_points={'console_scripts': ['labelme=labelme.__main__:main', 'labelme_draw_json=labelme.cli.draw_json:main', 'labelme_draw_label_png=labelme.cli.draw_label_png:main', 'labelme_json_to_dataset=labelme.cli.json_to_dataset:main', 'labelme_on_docker=labelme.cli.on_docker:main']}, data_files=[('share/man/man1', ['docs/man/labelme.1'])])

def get_long_description():
    with open('README.md') as f:
        long_description = f.read()
    try:
        import github2pypi
        return github2pypi.replace_url(slug='wkentaro/labelme', content=long_description)
    except Exception:
        return long_description

def voc_classes():
    return ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tvmonitor']

def cvt_to_coco_json(annotations):
    image_id = 0
    annotation_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    def addAnnItem(annotation_id, image_id, category_id, bbox, difficult_flag):
        annotation_item = dict()
        annotation_item['segmentation'] = []
        seg = []
        seg.append(int(bbox[0]))
        seg.append(int(bbox[1]))
        seg.append(int(bbox[0]))
        seg.append(int(bbox[3]))
        seg.append(int(bbox[2]))
        seg.append(int(bbox[3]))
        seg.append(int(bbox[2]))
        seg.append(int(bbox[1]))
        annotation_item['segmentation'].append(seg)
        xywh = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
        annotation_item['area'] = int(xywh[2] * xywh[3])
        if difficult_flag == 1:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 1
        else:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 0
        annotation_item['image_id'] = int(image_id)
        annotation_item['bbox'] = xywh.astype(int).tolist()
        annotation_item['category_id'] = int(category_id)
        annotation_item['id'] = int(annotation_id)
        coco['annotations'].append(annotation_item)
        return annotation_id + 1
    for category_id, name in enumerate(voc_classes()):
        category_item = dict()
        category_item['supercategory'] = str('none')
        category_item['id'] = int(category_id)
        category_item['name'] = str(name)
        coco['categories'].append(category_item)
    for ann_dict in annotations:
        file_name = ann_dict['filename']
        ann = ann_dict['ann']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(ann_dict['height'])
        image_item['width'] = int(ann_dict['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)
        bboxes = ann['bboxes'][:, :4]
        labels = ann['labels']
        for bbox_id in range(len(bboxes)):
            bbox = bboxes[bbox_id]
            label = labels[bbox_id]
            annotation_id = addAnnItem(annotation_id, image_id, label, bbox, difficult_flag=0)
        bboxes_ignore = ann['bboxes_ignore'][:, :4]
        labels_ignore = ann['labels_ignore']
        for bbox_id in range(len(bboxes_ignore)):
            bbox = bboxes_ignore[bbox_id]
            label = labels_ignore[bbox_id]
            annotation_id = addAnnItem(annotation_id, image_id, label, bbox, difficult_flag=1)
        image_id += 1
    return coco

def addAnnItem(annotation_id, image_id, category_id, bbox, difficult_flag):
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    seg.append(int(bbox[0]))
    seg.append(int(bbox[1]))
    seg.append(int(bbox[0]))
    seg.append(int(bbox[3]))
    seg.append(int(bbox[2]))
    seg.append(int(bbox[3]))
    seg.append(int(bbox[2]))
    seg.append(int(bbox[1]))
    annotation_item['segmentation'].append(seg)
    xywh = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
    annotation_item['area'] = int(xywh[2] * xywh[3])
    if difficult_flag == 1:
        annotation_item['ignore'] = 0
        annotation_item['iscrowd'] = 1
    else:
        annotation_item['ignore'] = 0
        annotation_item['iscrowd'] = 0
    annotation_item['image_id'] = int(image_id)
    annotation_item['bbox'] = xywh.astype(int).tolist()
    annotation_item['category_id'] = int(category_id)
    annotation_item['id'] = int(annotation_id)
    coco['annotations'].append(annotation_item)
    return annotation_id + 1

