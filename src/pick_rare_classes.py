import argparse
import json
from collections import Counter
from pathlib import Path

from utils import write_json


def count_boxes(data_root, split):
    path = Path(data_root) / 'annotations' / f'instances_{split}2017.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    id2label = {int(c['id']): c['name'] for c in data['categories']}
    counts = Counter()
    for ann in data['annotations']:
        counts[id2label[int(ann['category_id'])]] += 1
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--out-json', type=Path, default=Path('reports/class_distribution.json'))
    p.add_argument('--top-k-rare', type=int, default=3)
    args = p.parse_args()

    train = count_boxes(args.data_root, 'train')
    val = count_boxes(args.data_root, 'val')
    classes = sorted(set(train) | set(val))
    rows = []
    for c in classes:
        rows.append({'class': c, 'train_boxes': train.get(c, 0), 'val_boxes': val.get(c, 0)})
    rows.sort(key=lambda r: r['train_boxes'])
    rare = [r['class'] for r in rows[:args.top_k_rare]]

    write_json({'classes': rows, 'rare_classes_suggestion': rare}, args.out_json)
    print('rare classes:', ' '.join(rare))
    print('saved:', args.out_json)


if __name__ == '__main__':
    main()
