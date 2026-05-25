import argparse
import csv
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CocoDetrDataset, make_collate_fn
from modeling import get_processor, load_model_from_checkpoint
from utils import box_iou_xyxy, mkdir, write_json


def font(size=14):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def draw_example(img_path, out_path, gt, pred, id2label, title):
    img = Image.open(img_path).convert('RGB')
    d = ImageDraw.Draw(img)
    f = font(14)

    for box, label in zip(gt['boxes'], gt['labels']):
        x1, y1, x2, y2 = [float(x) for x in box]
        name = id2label.get(int(label), str(int(label)))
        d.rectangle([x1, y1, x2, y2], outline='green', width=3)
        d.text((x1, max(0, y1 - 16)), 'GT ' + name, fill='green', font=f)

    for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
        x1, y1, x2, y2 = [float(x) for x in box]
        name = id2label.get(int(label), str(int(label)))
        d.rectangle([x1, y1, x2, y2], outline='red', width=3)
        d.text((x1, min(img.height - 16, y2 + 2)), f'P {name} {float(score):.2f}', fill='red', font=f)

    d.rectangle([0, 0, min(img.width, 900), 24], fill='black')
    d.text((6, 4), title[:110], fill='white', font=f)
    mkdir(out_path.parent)
    img.save(out_path)


def analyse_one(pred, target, image_id, file_name, id2label, iou_thr, loc_min_iou):
    errors = []
    confusion = Counter()

    p_boxes, p_labels, p_scores = pred['boxes'], pred['labels'], pred['scores']
    t_boxes, t_labels = target['boxes'], target['labels']
    if len(p_boxes) == 0 and len(t_boxes) == 0:
        return errors, confusion

    if len(p_boxes) and len(t_boxes):
        ious = box_iou_xyxy(p_boxes, t_boxes)
    else:
        ious = torch.zeros((len(p_boxes), len(t_boxes)))

    matched_p = set()
    matched_t = set()

    # Greedy matching по IoU, чтобы один GT не засчитывался несколько раз.
    candidates = []
    for pi in range(len(p_boxes)):
        for gi in range(len(t_boxes)):
            candidates.append((float(ious[pi, gi]), pi, gi))
    candidates.sort(reverse=True)

    for iou, pi, gi in candidates:
        if iou < iou_thr:
            break
        if pi in matched_p or gi in matched_t:
            continue
        matched_p.add(pi)
        matched_t.add(gi)

        gt_label = int(t_labels[gi])
        pred_label = int(p_labels[pi])
        confusion[(gt_label, pred_label)] += 1
        if gt_label != pred_label:
            errors.append({
                'image_id': image_id,
                'file_name': file_name,
                'type': 'classification_error',
                'gt_label': id2label.get(gt_label, str(gt_label)),
                'pred_label': id2label.get(pred_label, str(pred_label)),
                'score': float(p_scores[pi]),
                'iou': round(iou, 4),
            })

    # Похожий box того же класса есть, но он не дотянул до IoU=0.5.
    for gi in range(len(t_boxes)):
        if gi in matched_t:
            continue
        same_class = [pi for pi in range(len(p_boxes))
                      if pi not in matched_p and int(p_labels[pi]) == int(t_labels[gi])]
        if not same_class:
            continue
        vals = ious[same_class, gi]
        best_iou, idx = torch.max(vals, dim=0)
        best_iou = float(best_iou)
        pi = same_class[int(idx)]
        if loc_min_iou <= best_iou < iou_thr:
            matched_p.add(pi)
            matched_t.add(gi)
            label = int(t_labels[gi])
            errors.append({
                'image_id': image_id,
                'file_name': file_name,
                'type': 'localization_error',
                'gt_label': id2label.get(label, str(label)),
                'pred_label': id2label.get(label, str(label)),
                'score': float(p_scores[pi]),
                'iou': round(best_iou, 4),
            })

    for gi in range(len(t_boxes)):
        if gi not in matched_t:
            label = int(t_labels[gi])
            errors.append({
                'image_id': image_id,
                'file_name': file_name,
                'type': 'missed_object',
                'gt_label': id2label.get(label, str(label)),
                'pred_label': '',
                'score': '',
                'iou': 0,
            })

    for pi in range(len(p_boxes)):
        if pi not in matched_p:
            label = int(p_labels[pi])
            best_iou = float(torch.max(ious[pi]).item()) if len(t_boxes) else 0.0
            errors.append({
                'image_id': image_id,
                'file_name': file_name,
                'type': 'false_positive',
                'gt_label': '',
                'pred_label': id2label.get(label, str(label)),
                'score': float(p_scores[pi]),
                'iou': round(best_iou, 4),
            })

    return errors, confusion


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--model-name', default=None)
    p.add_argument('--out-dir', type=Path, default=Path('reports/error_analysis'))
    p.add_argument('--split', default='val', choices=['train', 'val'])
    p.add_argument('--score-threshold', type=float, default=0.5)
    p.add_argument('--iou-threshold', type=float, default=0.5)
    p.add_argument('--loc-min-iou', type=float, default=0.1)
    p.add_argument('--num-examples', type=int, default=24)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    mkdir(args.out_dir / 'examples')
    device = torch.device(args.device)

    model_name = args.model_name or 'facebook/detr-resnet-50'
    processor = get_processor(model_name)
    model, id2label, _ = load_model_from_checkpoint(args.checkpoint, args.data_root, args.model_name)
    model.to(device).eval()

    ds = CocoDetrDataset(args.data_root, args.split, processor)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=make_collate_fn(processor))
    image_dir = args.data_root / 'images' / f'{args.split}2017'

    all_errors = []
    confusion = Counter()
    saved = 0

    for batch in tqdm(loader, desc='errors'):
        pixel_values = batch['pixel_values'].to(device)
        pixel_mask = batch['pixel_mask'].to(device) if batch['pixel_mask'] is not None else None
        target_sizes = batch['target_sizes'].to(device)
        out = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        results = processor.post_process_object_detection(out, threshold=args.score_threshold, target_sizes=target_sizes)

        for i, pred in enumerate(results):
            pred = {k: v.detach().cpu() for k, v in pred.items()}
            target = batch['metric_targets'][i]
            rows, conf = analyse_one(
                pred, target, batch['image_ids'][i], batch['file_names'][i],
                id2label, args.iou_threshold, args.loc_min_iou,
            )
            all_errors.extend(rows)
            confusion.update(conf)

            if rows and saved < args.num_examples:
                kind = rows[0]['type']
                out_img = args.out_dir / 'examples' / f'{saved:03d}_{kind}_{batch["file_names"][i]}'
                draw_example(image_dir / batch['file_names'][i], out_img, target, pred, id2label, kind)
                saved += 1

    with open(args.out_dir / 'errors.csv', 'w', newline='', encoding='utf-8') as f:
        cols = ['image_id', 'file_name', 'type', 'gt_label', 'pred_label', 'score', 'iou']
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_errors)

    with open(args.out_dir / 'confusion.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['gt_label', 'pred_label', 'count'])
        for (gt, pr), cnt in sorted(confusion.items()):
            w.writerow([id2label.get(gt, str(gt)), id2label.get(pr, str(pr)), cnt])

    summary = Counter(e['type'] for e in all_errors)
    write_json({'total_errors': len(all_errors), 'by_type': dict(summary)}, args.out_dir / 'summary.json')
    print('summary:', dict(summary))
    print('saved to:', args.out_dir)


if __name__ == '__main__':
    main()
