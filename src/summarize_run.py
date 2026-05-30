import argparse
import json
from pathlib import Path

import pandas as pd


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--reports-dir', type=Path, default=Path('reports'))
    p.add_argument('--out-md', type=Path, default=Path('reports/run_summary.md'))
    args = p.parse_args()

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    losses = read_csv(args.run_dir / 'train_losses.csv')
    metrics = read_csv(args.run_dir / 'val_metrics.csv')

    lines = []
    lines.append('# Run summary')
    lines.append('')
    lines.append(f'Run dir: `{args.run_dir}`')
    lines.append('')

    if metrics is not None and len(metrics):
        last = metrics.iloc[-1].to_dict()
        best_idx = metrics['map'].idxmax() if 'map' in metrics.columns else len(metrics) - 1
        best = metrics.loc[best_idx].to_dict()
        lines.append('## Validation metrics')
        lines.append('')
        lines.append(f"Last epoch: `{int(last.get('epoch', len(metrics)))}`")
        for k in ['map', 'map_50', 'map_75', 'mar_100']:
            if k in last:
                lines.append(f'- last `{k}`: `{float(last[k]):.4f}`')
        lines.append('')
        lines.append(f"Best epoch by mAP: `{int(best.get('epoch', best_idx + 1))}`")
        for k in ['map', 'map_50', 'map_75', 'mar_100']:
            if k in best:
                lines.append(f'- best `{k}`: `{float(best[k]):.4f}`')
        lines.append('')
    else:
        lines.append('No `val_metrics.csv` found yet.')
        lines.append('')

    if losses is not None and len(losses):
        lines.append('## Training loss')
        lines.append('')
        last_loss = losses.iloc[-1].to_dict()
        for k in ['loss_total', 'loss_ce', 'loss_bbox', 'loss_giou', 'cardinality_error']:
            if k in last_loss and pd.notna(last_loss[k]):
                lines.append(f'- last `{k}`: `{float(last_loss[k]):.4f}`')
        lines.append('')
        lines.append(f'Total logged steps: `{len(losses)}`')
        lines.append('')
    else:
        lines.append('No `train_losses.csv` found yet.')
        lines.append('')

    dist_path = args.reports_dir / 'class_distribution.json'
    if dist_path.exists():
        data = json.loads(dist_path.read_text())
        rare = data.get('rare_classes_suggestion', [])
        lines.append('## Class distribution')
        lines.append('')
        lines.append('Rare classes suggested for synthetic augmentation: ' + ', '.join(rare))
        lines.append('')

    expected = [
        args.run_dir / 'checkpoints' / 'best.pt',
        args.run_dir / 'tb',
        args.run_dir / 'profiler',
        args.reports_dir / 'figures',
        args.reports_dir / 'error_analysis',
        args.reports_dir / 'demo_report.html',
    ]
    lines.append('## Artifact checklist')
    lines.append('')
    for path in expected:
        mark = 'yes' if path.exists() else 'no'
        lines.append(f'- `{path}`: **{mark}**')

    args.out_md.write_text('\n'.join(lines), encoding='utf-8')
    print('saved:', args.out_md)


if __name__ == '__main__':
    main()
