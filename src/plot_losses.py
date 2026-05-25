import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils import mkdir


def plot(df, cols, out_file, title):
    plt.figure(figsize=(9, 4.8))
    for col in cols:
        if col in df.columns and df[col].notna().any():
            plt.plot(df['step'], df[col], label=col)
    plt.xlabel('step')
    plt.ylabel('value')
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=160)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log-csv', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, default=Path('reports/figures'))
    args = p.parse_args()

    mkdir(args.out_dir)
    df = pd.read_csv(args.log_csv)
    plot(df, ['loss_total'], args.out_dir / 'total_loss.png', 'DETR total loss')
    plot(df, ['loss_ce', 'loss_bbox', 'loss_giou', 'cardinality_error'],
         args.out_dir / 'detr_loss_components.png', 'DETR loss parts')
    print('saved plots to:', args.out_dir)


if __name__ == '__main__':
    main()
