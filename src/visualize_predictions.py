import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CocoDetrDataset, make_collate_fn
from modeling import get_processor, load_model_from_checkpoint
from utils import mkdir


def get_font():
    try:
        return ImageFont.truetype('DejaVuSans.ttf', 14)
    except OSError:
        return ImageFont.load_default()


def draw_boxes(img, boxes, labels, scores, id2label):
    d = ImageDraw.Draw(img)
    f = get_font()
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [float(v) for v in box]
        text = f"{id2label.get(int(label), str(int(label)))} {float(score):.2f}"
        d.rectangle([x1, y1, x2, y2], outline='red', width=3)
        tb = d.textbbox((x1, y1), text, font=f)
        d.rectangle(tb, fill='red')
        d.text((x1, y1), text, fill='white', font=f)
    return img


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--model-name', default=None)
    p.add_argument('--out-dir', type=Path, default=Path('reports/figures/predictions'))
    p.add_argument('--split', default='val', choices=['train', 'val'])
    p.add_argument('--num-images', type=int, default=24)
    p.add_argument('--score-threshold', type=float, default=0.5)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    mkdir(args.out_dir)
    device = torch.device(args.device)
    model_name = args.model_name or 'facebook/detr-resnet-50'
    processor = get_processor(model_name)
    model, id2label, _ = load_model_from_checkpoint(args.checkpoint, args.data_root, args.model_name)
    model.to(device).eval()

    ds = CocoDetrDataset(args.data_root, args.split, processor)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=make_collate_fn(processor))
    image_dir = args.data_root / 'images' / f'{args.split}2017'

    saved = 0
    for batch in tqdm(loader, desc='predictions'):
        pixel_values = batch['pixel_values'].to(device)
        pixel_mask = batch['pixel_mask'].to(device) if batch['pixel_mask'] is not None else None
        target_sizes = batch['target_sizes'].to(device)

        out = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        results = processor.post_process_object_detection(out, threshold=args.score_threshold, target_sizes=target_sizes)

        for file_name, pred in zip(batch['file_names'], results):
            img = Image.open(image_dir / file_name).convert('RGB')
            img = draw_boxes(img, pred['boxes'].cpu(), pred['labels'].cpu(), pred['scores'].cpu(), id2label)
            img.save(args.out_dir / file_name)
            saved += 1
            if saved >= args.num_images:
                print('saved:', saved, '->', args.out_dir)
                return
    print('saved:', saved, '->', args.out_dir)


if __name__ == '__main__':
    main()
