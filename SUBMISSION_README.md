# CV HW3 submission package

Main files:

- `src/` — source code for DETR training, visualization, error analysis and synthetic ablation.
- `README.md` — experiment instructions.
- `reports/demo_report.html` — full visual report.
- `reports/HW3_final_summary.md` — compact final summary.
- `runs/detr_coco10/val_metrics.csv` — DETR validation metrics.
- `runs/detr_coco10/train_losses.csv` — DETR training losses.
- `runs/detr_coco10/tb/` — TensorBoard logs, if present.
- `runs/detr_coco10/profiler/` — profiler trace, if present.
- `runs/synthetic_ablation/ablation_metrics.csv` — real-only vs real+synthetic ablation.
- `reports/figures/predictions/` — prediction gallery.
- `reports/error_analysis/` — error analysis outputs.
- `reports/figures/synthetic_examples/` — synthetic examples.

Large raw datasets and model checkpoints are not included in Git.

- `notebooks/hw3_detr_coco_synthetic_colab.ipynb` — final Colab notebook with Drive persistence, runtime recovery, training, visualization, error analysis, synthetic ablation and report packaging.
