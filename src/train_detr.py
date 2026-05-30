import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from dataset import CocoDetrDataset, make_collate_fn
from modeling import get_model, get_processor
from utils import EmptyProfiler, append_csv, as_float, mkdir, set_seed


TRAIN_COLUMNS = ['epoch', 'step', 'loss_total', 'loss_ce', 'loss_bbox', 'loss_giou', 'cardinality_error']
VAL_COLUMNS = ['epoch', 'map', 'map_50', 'map_75', 'mar_100']


def make_optimizer(model, lr, lr_backbone, weight_decay):
    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    return torch.optim.AdamW(
        [
            {'params': head_params, 'lr': lr},
            {'params': backbone_params, 'lr': lr_backbone},
        ],
        weight_decay=weight_decay,
    )




def move_labels_to_device(labels, device):
    # HuggingFace DETR expects a list of dictionaries.
    # Each tensor inside every target must be on the same device as pixel_values.
    moved = []
    for target in labels:
        moved.append({
            key: value.to(device) if hasattr(value, 'to') else value
            for key, value in target.items()
        })
    return moved

def make_profiler(args, device, enabled):
    if not enabled:
        return EmptyProfiler()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=args.profile_steps, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(args.output_dir / 'profiler')),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )


@torch.no_grad()
def evaluate(model, loader, processor, device, score_thr):
    model.eval()
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)

    for batch in tqdm(loader, desc='val', leave=False):
        pixel_values = batch['pixel_values'].to(device)
        pixel_mask = batch['pixel_mask'].to(device) if batch['pixel_mask'] is not None else None
        target_sizes = batch['target_sizes'].to(device)

        out = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        pred = processor.post_process_object_detection(out, threshold=score_thr, target_sizes=target_sizes)

        preds = []
        for p in pred:
            preds.append({
                'boxes': p['boxes'].detach().cpu(),
                'scores': p['scores'].detach().cpu(),
                'labels': p['labels'].detach().cpu(),
            })
        targets = []
        for t in batch['metric_targets']:
            targets.append({'boxes': t['boxes'].cpu(), 'labels': t['labels'].cpu()})
        metric.update(preds, targets)

    raw = metric.compute()
    return {k: as_float(raw[k]) for k in ['map', 'map_50', 'map_75', 'mar_100'] if k in raw}


def save_ckpt(path, model, optim, epoch, args, metrics):
    mkdir(path.parent)
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optim.state_dict(),
        'args': vars(args),
        'metrics': metrics,
    }, path)


def train_epoch(model, loader, optim, writer, args, device, epoch, global_step):
    model.train()
    sums = {}
    counts = {}
    profile_this_epoch = args.profile and epoch == 1

    with make_profiler(args, device, profile_this_epoch) as prof:
        pbar = tqdm(loader, desc=f'train {epoch}')
        for batch in pbar:
            labels = move_labels_to_device(batch['labels'], device)
            pixel_values = batch['pixel_values'].to(device)
            pixel_mask = batch['pixel_mask'].to(device) if batch['pixel_mask'] is not None else None

            out = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = out.loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()

            row = {'epoch': epoch, 'step': global_step, 'loss_total': float(loss.detach().cpu())}
            writer.add_scalar('train/loss_total', row['loss_total'], global_step)

            loss_dict = getattr(out, 'loss_dict', {}) or {}
            for name, val in loss_dict.items():
                row[name] = float(val.detach().cpu())
                writer.add_scalar('train/' + name, row[name], global_step)

            append_csv(args.output_dir / 'train_losses.csv', row, TRAIN_COLUMNS)

            for k, v in row.items():
                if k in ('epoch', 'step'):
                    continue
                sums[k] = sums.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1

            pbar.set_postfix(loss=f"{row['loss_total']:.3f}")
            global_step += 1
            prof.step()

    means = {k: sums[k] / max(counts[k], 1) for k in sums}
    return global_step, means


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--model-name', default='facebook/detr-resnet-50')
    p.add_argument('--output-dir', type=Path, default=Path('runs/detr_coco10'))
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--lr-backbone', type=float, default=1e-6)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--grad-clip', type=float, default=0.1)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--score-threshold', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--resume', type=Path, default=None,
                   help='path to checkpoint .pt; useful when Colab runtime dies')
    p.add_argument('--profile-steps', type=int, default=5)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    set_seed(args.seed)
    mkdir(args.output_dir / 'checkpoints')
    device = torch.device(args.device)

    processor = get_processor(args.model_name)
    model, id2label, label2id = get_model(args.model_name, args.data_root)
    model.to(device)

    train_ds = CocoDetrDataset(args.data_root, 'train', processor)
    val_ds = CocoDetrDataset(args.data_root, 'val', processor)
    collate = make_collate_fn(processor)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate,
                              pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate,
                            pin_memory=(device.type == 'cuda'))

    optim = make_optimizer(model, args.lr, args.lr_backbone, args.weight_decay)
    writer = SummaryWriter(args.output_dir / 'tb')

    best_map = -1.0
    global_step = 0
    start_epoch = 1

    if args.resume is not None and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        if 'optimizer_state' in ckpt:
            optim.load_state_dict(ckpt['optimizer_state'])
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        best_map = float(ckpt.get('metrics', {}).get('map', -1.0))
        print('resumed from:', args.resume)
        print('start epoch:', start_epoch)
    print('device:', device)
    print('train/val:', len(train_ds), len(val_ds))
    print('labels:', id2label)

    for epoch in range(start_epoch, args.epochs + 1):
        global_step, train_means = train_epoch(model, train_loader, optim, writer, args, device, epoch, global_step)
        for k, v in train_means.items():
            writer.add_scalar('epoch_train/' + k, v, epoch)

        metrics = evaluate(model, val_loader, processor, device, args.score_threshold)
        append_csv(args.output_dir / 'val_metrics.csv', {'epoch': epoch, **metrics}, VAL_COLUMNS)
        for k, v in metrics.items():
            writer.add_scalar('val/' + k, v, epoch)

        epoch_ckpt = args.output_dir / 'checkpoints' / f'epoch_{epoch:03d}.pt'
        save_ckpt(epoch_ckpt, model, optim, epoch, args, metrics)
        # latest.pt is convenient for Colab resume after runtime disconnects.
        save_ckpt(args.output_dir / 'checkpoints' / 'latest.pt', model, optim, epoch, args, metrics)
        if metrics.get('map', -1.0) > best_map:
            best_map = metrics.get('map', -1.0)
            save_ckpt(args.output_dir / 'checkpoints' / 'best.pt', model, optim, epoch, args, metrics)

        print(f'epoch {epoch}:', metrics)

    writer.close()
    print('best mAP:', round(best_map, 4))
    print('artifacts:', args.output_dir)


if __name__ == '__main__':
    main()
