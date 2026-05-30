# HW3 Final Summary: DETR + Synthetic Data Ablation

## Part 1: DETR on COCO-subset

Dataset: COCO-subset, 10 classes.

Classes:
person, bicycle, car, motorcycle, bus, truck, cat, dog, bottle, chair.

Train images: 1200  
Validation images: 300  
Model: facebook/detr-resnet-50  
Epochs target: 30  
Batch size: 2  
Learning rate: 1e-5  
Backbone learning rate: 1e-6  

Final validation metrics:

|   epoch |      map |   map_50 |   map_75 |   mar_100 |
|--------:|---------:|---------:|---------:|----------:|
|      30 | 0.355334 | 0.523353 | 0.369194 |  0.422141 |

Best validation checkpoint by mAP:

```json
{
  "epoch": 30.0,
  "map": 0.3553341329097748,
  "map_50": 0.5233533978462219,
  "map_75": 0.3691941499710083,
  "mar_100": 0.4221411645412445
}
```

Main observations:
- DETR quality grows during training, especially after the first epochs.
- Early epochs can have near-zero mAP; this is expected for DETR fine-tuning on a small subset.
- mAP50 is higher than mAP because IoU=0.50 is a softer localization criterion.
- Error analysis is included to show false positives, missed objects, localization errors and classification errors.

Error-analysis summary:

```json
{
  "total_errors": 1271,
  "by_type": {
    "missed_object": 154,
    "false_positive": 976,
    "localization_error": 119,
    "classification_error": 22
  }
}
```

## Part 2: Synthetic Data with Stable Diffusion + ControlNet

Rare classes selected for synthetic augmentation:
bus, cat, bicycle

Pipeline:
1. Object crops were extracted from COCO bounding boxes.
2. Synthetic images were generated for rare classes using Stable Diffusion + ControlNet.
3. A classification ablation was run:
   - baseline: real crops only;
   - experiment: real crops + synthetic images.
4. Metrics were compared to estimate whether synthetic augmentation helped.

Ablation table:

| experiment          | train_data              |   epoch |   accuracy |   macro_f1 |
|:--------------------|:------------------------|--------:|-----------:|-----------:|
| real_only           | real crops only         |       2 |   0.848577 |   0.696259 |
| real_plus_synthetic | real crops + ControlNet |       4 |   0.82622  |   0.734213 |

Interpretation:
- Accuracy delta real+synthetic vs real-only: -0.0224.
- Macro-F1 delta real+synthetic vs real-only: +0.0380.
- If macro-F1 improves while accuracy slightly decreases, synthetic data is best interpreted as helping class balance rather than improving every aggregate metric.

Conclusion:
Synthetic data is treated as an experimental augmentation, not as guaranteed improvement. The effect is interpreted through the ablation table and qualitative examples.

## Saved artifacts

- DETR checkpoints: `runs/detr_coco10/checkpoints/`
- TensorBoard logs: `runs/detr_coco10/tb/`
- Profiler trace: `runs/detr_coco10/profiler/`
- DETR metrics: `runs/detr_coco10/val_metrics.csv`
- Loss logs: `runs/detr_coco10/train_losses.csv`
- Prediction gallery: `reports/figures/predictions/`
- Error analysis: `reports/error_analysis/`
- Synthetic examples: `reports/figures/synthetic_examples/`
- Synthetic ablation: `runs/synthetic_ablation/` and `reports/synthetic_ablation_metrics.csv`
