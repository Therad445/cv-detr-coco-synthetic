import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import xywh_to_xyxy


class CocoDetrDataset(Dataset):
    # Небольшая обертка вокруг COCO-json. Оставляю формат максимально близко к COCO,
    # потому что DetrImageProcessor сам делает нормализацию bbox для loss.
    def __init__(self, root, split, processor):
        self.root = Path(root)
        self.split = split
        self.processor = processor
        self.ann_file = self.root / 'annotations' / f'instances_{split}2017.json'
        self.img_dir = self.root / 'images' / f'{split}2017'

        with open(self.ann_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.images = sorted(data['images'], key=lambda x: x['id'])
        self.categories = {int(c['id']): c['name'] for c in data['categories']}

        anns = defaultdict(list)
        for a in data['annotations']:
            if a.get('iscrowd', 0):
                continue
            x, y, w, h = a['bbox']
            if w <= 1 or h <= 1:
                continue
            anns[int(a['image_id'])].append(dict(a))
        self.anns = anns

    def __len__(self):
        return len(self.images)

    def _target_for_map(self, anns):
        boxes, labels = [], []
        for a in anns:
            boxes.append(xywh_to_xyxy(a['bbox']))
            labels.append(int(a['category_id']))
        if not boxes:
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.long),
            }
        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
        }

    def __getitem__(self, idx):
        info = self.images[idx]
        image_id = int(info['id'])
        image = Image.open(self.img_dir / info['file_name']).convert('RGB')
        anns = self.anns.get(image_id, [])

        encoded = self.processor(
            images=image,
            annotations={'image_id': image_id, 'annotations': anns},
            return_tensors='pt',
        )

        return {
            'pixel_values': encoded['pixel_values'].squeeze(0),
            'labels': encoded['labels'][0],
            'metric_target': self._target_for_map(anns),
            'target_size': torch.tensor([image.height, image.width], dtype=torch.long),
            'image_id': image_id,
            'file_name': info['file_name'],
        }


def make_collate_fn(processor):
    def collate(batch):
        padded = processor.pad([b['pixel_values'] for b in batch], return_tensors='pt')
        return {
            'pixel_values': padded['pixel_values'],
            'pixel_mask': padded.get('pixel_mask'),
            'labels': [b['labels'] for b in batch],
            'metric_targets': [b['metric_target'] for b in batch],
            'target_sizes': torch.stack([b['target_size'] for b in batch]),
            'image_ids': [b['image_id'] for b in batch],
            'file_names': [b['file_name'] for b in batch],
        }
    return collate
