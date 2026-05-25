import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from utils import mkdir, write_json


def extract_split(data_root, out_root, split, min_area, max_per_class):
    ann_file = data_root / 'annotations' / f'instances_{split}2017.json'
    img_dir = data_root / 'images' / f'{split}2017'
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    id2label = {int(c['id']): c['name'] for c in data['categories']}
    images = {int(img['id']): img for img in data['images']}
    by_image = defaultdict(list)

    for ann in data['annotations']:
        x, y, w, h = ann['bbox']
        if w * h >= min_area:
            by_image[int(ann['image_id'])].append(ann)

    counts = Counter()
    rows = []
    for image_id, anns in tqdm(by_image.items(), desc=f'crop {split}'):
        info = images[image_id]
        path = img_dir / info['file_name']
        if not path.exists():
            continue
        img = Image.open(path).convert('RGB')

        for ann in anns:
            cls = id2label[int(ann['category_id'])]
            if max_per_class > 0 and counts[cls] >= max_per_class:
                continue
            x, y, w, h = ann['bbox']
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(img.width, int(x + w)), min(img.height, int(y + h))
            if x2 <= x1 or y2 <= y1:
                continue

            out_dir = out_root / split / cls
            mkdir(out_dir)
            out_name = f'{image_id}_{ann["id"]}.jpg'
            img.crop((x1, y1, x2, y2)).save(out_dir / out_name, quality=95)
            counts[cls] += 1
            rows.append({'split': split, 'class': cls, 'file': str(out_dir / out_name), 'source': info['file_name']})

    return counts, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--min-area', type=int, default=1024)
    p.add_argument('--max-per-class', type=int, default=700)
    args = p.parse_args()

    all_rows = []
    stats = {}
    for split in ['train', 'val']:
        counts, rows = extract_split(args.data_root, args.out_root, split, args.min_area, args.max_per_class)
        all_rows.extend(rows)
        stats[split] = dict(counts)
        print(split, dict(counts))
    write_json(stats, args.out_root / 'crop_stats.json')
    print('saved to:', args.out_root)


if __name__ == '__main__':
    main()
