import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from utils import mkdir, write_json


CLASSES = ['red_square', 'blue_circle', 'green_triangle']


def draw_one(path, cls_id, idx):
    img = Image.new('RGB', (256, 256), 'white')
    d = ImageDraw.Draw(img)
    x1 = 25 + (idx * 17) % 120
    y1 = 30 + (idx * 23) % 110
    x2 = x1 + 70
    y2 = y1 + 70
    if cls_id == 0:
        d.rectangle([x1, y1, x2, y2], fill='red')
    elif cls_id == 1:
        d.ellipse([x1, y1, x2, y2], fill='blue')
    else:
        d.polygon([(x1 + 35, y1), (x1, y2), (x2, y2)], fill='green')
    img.save(path)
    return [x1, y1, x2 - x1, y2 - y1]


def make_split(root, split, n):
    img_dir = root / 'images' / f'{split}2017'
    mkdir(img_dir)
    images, anns = [], []
    ann_id = 1
    for i in range(n):
        cls_id = i % len(CLASSES)
        file_name = f'{split}_{i:04d}.jpg'
        bbox = draw_one(img_dir / file_name, cls_id, i)
        images.append({'id': i + 1, 'file_name': file_name, 'width': 256, 'height': 256})
        anns.append({
            'id': ann_id,
            'image_id': i + 1,
            'category_id': cls_id,
            'bbox': bbox,
            'area': bbox[2] * bbox[3],
            'iscrowd': 0,
        })
        ann_id += 1
    cats = [{'id': i, 'name': name, 'supercategory': 'toy'} for i, name in enumerate(CLASSES)]
    out = {'images': images, 'annotations': anns, 'categories': cats, 'info': {}, 'licenses': []}
    mkdir(root / 'annotations')
    with open(root / 'annotations' / f'instances_{split}2017.json', 'w', encoding='utf-8') as f:
        json.dump(out, f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-root', type=Path, default=Path('data/toy_coco'))
    p.add_argument('--train-size', type=int, default=36)
    p.add_argument('--val-size', type=int, default=12)
    args = p.parse_args()
    make_split(args.out_root, 'train', args.train_size)
    make_split(args.out_root, 'val', args.val_size)
    write_json({
        'id2label': {str(i): name for i, name in enumerate(CLASSES)},
        'label2id': {name: i for i, name in enumerate(CLASSES)},
    }, args.out_root / 'annotations' / 'id2label.json')
    print('toy dataset:', args.out_root)


if __name__ == '__main__':
    main()
