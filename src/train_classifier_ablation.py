import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
from tqdm import tqdm

from utils import mkdir, set_seed


def tfm(train):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.15, 0.15, 0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def make_model(num_classes):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_exp(name, train_ds, val_ds, class_names, args):
    device = torch.device(args.device)
    writer = SummaryWriter(args.output_dir / 'tb' / name)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = make_model(len(class_names)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = {'experiment': name, 'epoch': 0, 'accuracy': 0.0, 'macro_f1': 0.0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in tqdm(train_loader, desc=f'{name} train {epoch}', leave=False):
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        writer.add_scalar('train/loss', float(np.mean(losses)) if losses else 0.0, epoch)

        model.eval()
        pred, true = [], []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f'{name} val {epoch}', leave=False):
                logits = model(x.to(device))
                pred.extend(torch.argmax(logits, dim=1).cpu().tolist())
                true.extend(y.tolist())
        acc = accuracy_score(true, pred)
        f1 = f1_score(true, pred, average='macro', zero_division=0)
        writer.add_scalar('val/accuracy', acc, epoch)
        writer.add_scalar('val/macro_f1', f1, epoch)
        print(f'{name} epoch {epoch}: acc={acc:.4f} macro_f1={f1:.4f}')

        if f1 > best['macro_f1']:
            best = {'experiment': name, 'epoch': epoch, 'accuracy': acc, 'macro_f1': f1}
            torch.save({'model_state': model.state_dict(), 'classes': class_names, 'metrics': best},
                       args.output_dir / f'{name}_best.pt')
    writer.close()
    return best


def write_rows(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        cols = ['experiment', 'train_data', 'epoch', 'accuracy', 'macro_f1']
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--real-root', type=Path, required=True, help='data/crops10')
    p.add_argument('--synthetic-root', type=Path, required=True, help='ImageFolder with synthetic class dirs')
    p.add_argument('--output-dir', type=Path, default=Path('runs/synthetic_ablation'))
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    set_seed(args.seed)
    mkdir(args.output_dir)

    real_train = datasets.ImageFolder(args.real_root / 'train', transform=tfm(True))
    real_val = datasets.ImageFolder(args.real_root / 'val', transform=tfm(False))
    synth = datasets.ImageFolder(args.synthetic_root, transform=tfm(True))

    if real_train.classes != real_val.classes:
        raise ValueError('train/val folders have different classes')
    if synth.classes != real_train.classes:
        raise ValueError(f'synthetic classes must match real classes: {synth.classes} vs {real_train.classes}')

    rows = []
    r1 = run_exp('real_only', real_train, real_val, real_train.classes, args)
    r1['train_data'] = 'real crops only'
    rows.append(r1)

    r2 = run_exp('real_plus_synthetic', ConcatDataset([real_train, synth]), real_val, real_train.classes, args)
    r2['train_data'] = 'real crops + ControlNet'
    rows.append(r2)

    write_rows(args.output_dir / 'ablation_metrics.csv', rows)
    print('saved:', args.output_dir / 'ablation_metrics.csv')


if __name__ == '__main__':
    main()
