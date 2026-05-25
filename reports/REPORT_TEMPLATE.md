# Отчёт по CV HW

## 1. Постановка

Цель — обучить DETR на небольшом COCO-subset и проверить, помогает ли синтетическая аугментация для редких классов.

## 2. Данные

Классы:

```text
person, bicycle, car, motorcycle, bus, truck, cat, dog, chair, bottle
```

Размер subset:

| split | images | boxes |
|---|---:|---:|
| train | TODO | TODO |
| val | TODO | TODO |

Редкие классы выбирались по числу bbox в train split. Файл: `reports/class_distribution.json`.

## 3. Обучение DETR

Модель: `facebook/detr-resnet-50`.

| parameter | value |
|---|---:|
| epochs | 5 |
| batch size | 2 |
| lr head | 1e-5 |
| lr backbone | 1e-6 |
| weight decay | 1e-4 |
| score threshold | 0.5 |

## 4. Метрики

| experiment | mAP | mAP50 | mAP75 | mAR100 |
|---|---:|---:|---:|---:|
| DETR fine-tuning | TODO | TODO | TODO | TODO |

Числа берутся из `runs/detr_coco10/val_metrics.csv`.

## 5. Loss curves

Вставить:

- `reports/figures/total_loss.png`
- `reports/figures/detr_loss_components.png`

Короткий комментарий:

- TODO: total loss снижается / не снижается;
- TODO: bbox и giou ведут себя стабильно / нестабильно;
- TODO: classification loss показывает ...

## 6. Qualitative examples

Вставить 4–8 картинок из `reports/figures/predictions/`.

## 7. Error analysis

Файлы:

- `reports/error_analysis/summary.json`
- `reports/error_analysis/errors.csv`
- `reports/error_analysis/confusion.csv`
- `reports/error_analysis/examples/`

| error type | count |
|---|---:|
| classification_error | TODO |
| localization_error | TODO |
| missed_object | TODO |
| false_positive | TODO |

Вывод:

- TODO: больше всего ошибок типа ...
- TODO: локализация чаще ломается на ...
- TODO: классы, которые путаются чаще всего: ...

## 8. Синтетика и ablation

Генерация: Stable Diffusion 1.5 + ControlNet Canny. Control image строится из crop-а реального объекта через Canny edge map.

| experiment | accuracy | macro F1 |
|---|---:|---:|
| real only | TODO | TODO |
| real + synthetic | TODO | TODO |

Числа берутся из `runs/synthetic_ablation/ablation_metrics.csv`.

Вывод:

- TODO: синтетика помогла / не помогла;
- TODO: вероятная причина;
- TODO: что можно улучшить дальше.

## 9. Что можно улучшить

- Запустить больше эпох и добавить scheduler.
- Попробовать Deformable-DETR как более быстрый вариант для малых объектов.
- Для синтетики фильтровать плохие изображения вручную или CLIP-score/простым classifier sanity check.
- Делать синтетику не только на crop-ах, но и вставлять объекты в реальные сцены.
