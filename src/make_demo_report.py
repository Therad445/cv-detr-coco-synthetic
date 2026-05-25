import argparse
import html
import json
from pathlib import Path

import pandas as pd

from utils import mkdir


def img_grid(paths, rel_root, limit=16):
    items = []
    for p in paths[:limit]:
        rel = p.relative_to(rel_root).as_posix()
        items.append(f'<figure><img src="{html.escape(rel)}"><figcaption>{html.escape(p.name)}</figcaption></figure>')
    return '\n'.join(items) if items else '<p class="muted">Пока нет картинок. Запусти visualize/error_analysis.</p>'


def table_from_csv(path):
    if not path.exists():
        return '<p class="muted">Файл не найден: ' + html.escape(str(path)) + '</p>'
    df = pd.read_csv(path)
    return df.tail(10).to_html(index=False, classes='tbl', border=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, default=Path('runs/detr_coco10'))
    p.add_argument('--reports-dir', type=Path, default=Path('reports'))
    p.add_argument('--out-file', type=Path, default=Path('reports/demo_report.html'))
    args = p.parse_args()

    mkdir(args.out_file.parent)
    figures = args.reports_dir / 'figures'
    pred_dir = figures / 'predictions'
    err_dir = args.reports_dir / 'error_analysis'

    pred_imgs = sorted([p for p in pred_dir.glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    err_imgs = sorted([p for p in (err_dir / 'examples').glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])

    summary_text = '{}'
    summary_path = err_dir / 'summary.json'
    if summary_path.exists():
        summary_text = json.dumps(json.loads(summary_path.read_text(encoding='utf-8')), ensure_ascii=False, indent=2)

    # Пути в html делаю относительно папки reports/, чтобы файл можно было открыть двойным кликом.
    rel_root = args.reports_dir
    html_doc = f'''<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>CV HW demo</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 36px; line-height: 1.45; color: #222; }}
h1, h2 {{ margin-bottom: 8px; }}
.card {{ border: 1px solid #ddd; border-radius: 12px; padding: 18px; margin: 18px 0; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 14px; }}
figure {{ margin: 0; border: 1px solid #eee; border-radius: 10px; padding: 8px; }}
img {{ max-width: 100%; border-radius: 8px; }}
figcaption {{ font-size: 12px; color: #666; margin-top: 6px; word-break: break-all; }}
.tbl {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
.tbl th, .tbl td {{ border-bottom: 1px solid #eee; padding: 6px 8px; text-align: left; }}
.muted {{ color: #777; }}
pre {{ background: #f7f7f7; padding: 12px; border-radius: 8px; overflow-x: auto; }}
</style>
</head>
<body>
<h1>DETR / COCO-subset: демонстрационный отчёт</h1>
<p class="muted">Этот файл собирается автоматически из логов, графиков и картинок проекта.</p>

<div class="card">
<h2>Метрики валидации</h2>
{table_from_csv(args.run_dir / 'val_metrics.csv')}
</div>

<div class="card">
<h2>Loss curves</h2>
<div class="grid">
<figure><img src="figures/total_loss.png"><figcaption>total loss</figcaption></figure>
<figure><img src="figures/detr_loss_components.png"><figcaption>loss components</figcaption></figure>
</div>
</div>

<div class="card">
<h2>Предсказанные боксы</h2>
<div class="grid">{img_grid(pred_imgs, rel_root)}</div>
</div>

<div class="card">
<h2>Error analysis</h2>
<pre>{html.escape(summary_text)}</pre>
<div class="grid">{img_grid(err_imgs, rel_root)}</div>
</div>
</body>
</html>'''
    args.out_file.write_text(html_doc, encoding='utf-8')
    print('saved:', args.out_file)


if __name__ == '__main__':
    main()
