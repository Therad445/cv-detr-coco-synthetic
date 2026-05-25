import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from utils import mkdir, write_json


DEFAULT_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus',
    'truck', 'cat', 'dog', 'chair', 'bottle',
]


def copy_image(src, dst, mode):
    if mode == 'none' or dst.exists():
        return
    mkdir(dst.parent)
    if mode == 'copy':
        shutil.copy2(src, dst)
        return
    if mode == 'symlink':
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)
        return
    raise ValueError(f'bad link mode: {mode}')


def make_split(coco_root, out_root, split, class_names, max_images, link_mode):
    ann_file = coco_root / 'annotations' / f'instances_{split}2017.json'
    img_src = coco_root / f'{split}2017'
    img_dst = out_root / 'images' / f'{split}2017'

    with open(ann_file, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    name_to_old_id = {c['name']: int(c['id']) for c in coco['categories']}
    missing = [c for c in class_names if c not in name_to_old_id]
    if missing:
        raise ValueError(f'No such COCO classes: {missing}')

    old_to_new = {name_to_old_id[name]: i for i, name in enumerate(class_names)}
    wanted_old_ids = set(old_to_new)

    by_image = defaultdict(list)
    for ann in coco['annotations']:
        if ann.get('iscrowd', 0):
            continue
        if int(ann['category_id']) not in wanted_old_ids:
            continue
        x, y, w, h = ann['bbox']
        if w <= 1 or h <= 1:
            continue
        by_image[int(ann['image_id'])].append(ann)

    images = [img for img in coco['images'] if int(img['id']) in by_image]
    images = sorted(images, key=lambda x: int(x['id']))
    if max_images > 0:
        images = images[:max_images]

    new_images = []
    new_annotations = []
    new_ann_id = 1

    for img in tqdm(images, desc=f'{split} images'):
        src = img_src / img['file_name']
        dst = img_dst / img['file_name']
        if not src.exists():
            raise FileNotFoundError(src)
        copy_image(src, dst, link_mode)
        new_images.append(img)

        for ann in by_image[int(img['id'])]:
            ann = dict(ann)
            ann['id'] = new_ann_id
            ann['category_id'] = old_to_new[int(ann['category_id'])]
            new_annotations.append(ann)
            new_ann_id += 1

    new_categories = [
        {'id': i, 'name': name, 'supercategory': 'object'}
        for i, name in enumerate(class_names)
    ]

    subset = {
        'info': {'description': f'COCO {split}2017 subset', 'classes': class_names},
        'licenses': coco.get('licenses', []),
        'images': new_images,
        'annotations': new_annotations,
        'categories': new_categories,
    }
    mkdir(out_root / 'annotations')
    with open(out_root / 'annotations' / f'instances_{split}2017.json', 'w', encoding='utf-8') as f:
        json.dump(subset, f, ensure_ascii=False)
    return subset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--coco-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--classes', nargs='+', default=DEFAULT_CLASSES)
    p.add_argument('--max-train-images', type=int, default=1200)
    p.add_argument('--max-val-images', type=int, default=300)
    p.add_argument('--link-mode', choices=['copy', 'symlink', 'none'], default='copy')
    args = p.parse_args()

    train = make_split(args.coco_root, args.out_root, 'train', args.classes, args.max_train_images, args.link_mode)
    val = make_split(args.coco_root, args.out_root, 'val', args.classes, args.max_val_images, args.link_mode)

    id2label = {str(c['id']): c['name'] for c in train['categories']}
    label2id = {name: int(idx) for idx, name in id2label.items()}
    write_json({'id2label': id2label, 'label2id': label2id}, args.out_root / 'annotations' / 'id2label.json')

    print('saved to:', args.out_root)
    print('train:', len(train['images']), 'images /', len(train['annotations']), 'boxes')
    print('val:  ', len(val['images']), 'images /', len(val['annotations']), 'boxes')
    print('classes:', ', '.join(args.classes))


if __name__ == '__main__':
    main()
