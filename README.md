# HW: DETR на COCO-subset и синтетика через ControlNet

В проекте две части.

1. Детектор: COCO-subset на 10 классов, fine-tuning DETR, TensorBoard, чекпойнты, profiler trace, mAP/mAP50, графики loss и разбор ошибок.
2. Синтетика: crop-ы объектов из COCO, генерация дополнительных примеров через Stable Diffusion + ControlNet, сравнение CNN без синтетики и с синтетикой.

Большие файлы сюда не кладу: COCO, чекпойнты, TensorBoard-логи и картинки генерируются командами ниже.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Для нормального обучения нужен GPU. На CPU имеет смысл запускать только маленький smoke-test.

## Данные

Ожидаю обычную структуру COCO 2017:

```text
data/coco/
  train2017/
  val2017/
  annotations/
    instances_train2017.json
    instances_val2017.json
```

## 1. COCO-subset

```bash
python src/prepare_coco_subset.py \
  --coco-root data/coco \
  --out-root data/coco10 \
  --classes person bicycle car motorcycle bus truck cat dog chair bottle \
  --max-train-images 1200 \
  --max-val-images 300 \
  --link-mode copy
```

В `data/coco10/annotations/id2label.json` классы ремапятся в `0..9`, чтобы голова DETR обучалась сразу на нужное число классов.

Проверка распределения классов:

```bash
python src/pick_rare_classes.py --data-root data/coco10 --out-json reports/class_distribution.json
```

Этот файл полезно вставить в отчёт: видно, какие классы редкие и почему именно их беру для синтетики.

## 2. Fine-tuning DETR

```bash
python src/train_detr.py \
  --data-root data/coco10 \
  --model-name facebook/detr-resnet-50 \
  --output-dir runs/detr_coco10 \
  --epochs 5 \
  --batch-size 2 \
  --lr 1e-5 \
  --lr-backbone 1e-6 \
  --weight-decay 1e-4 \
  --num-workers 4 \
  --profile
```

После запуска должны появиться:

```text
runs/detr_coco10/
  tb/
  profiler/
  train_losses.csv
  val_metrics.csv
  checkpoints/best.pt
```

TensorBoard:

```bash
tensorboard --logdir runs/detr_coco10/tb
```

## 3. Графики loss

```bash
python src/plot_losses.py \
  --log-csv runs/detr_coco10/train_losses.csv \
  --out-dir reports/figures
```

## 4. Визуализация предсказаний

```bash
python src/visualize_predictions.py \
  --data-root data/coco10 \
  --checkpoint runs/detr_coco10/checkpoints/best.pt \
  --out-dir reports/figures/predictions \
  --num-images 24 \
  --score-threshold 0.5
```

## 5. Error analysis

```bash
python src/error_analysis.py \
  --data-root data/coco10 \
  --checkpoint runs/detr_coco10/checkpoints/best.pt \
  --out-dir reports/error_analysis \
  --score-threshold 0.5 \
  --iou-threshold 0.5
```

Скрипт разделяет ошибки на четыре группы:

- `classification_error` — box попал в объект, но класс неверный;
- `localization_error` — класс верный, но IoU ниже 0.5;
- `missed_object` — объект в разметке есть, а предсказания нет;
- `false_positive` — модель увидела лишний объект.

## 6. HTML-демо для защиты

После обучения и визуализаций:

```bash
python src/make_demo_report.py \
  --run-dir runs/detr_coco10 \
  --reports-dir reports \
  --out-file reports/demo_report.html
```

Открой `reports/demo_report.html` в браузере. Там будут последние метрики, графики loss, примеры предсказаний и галерея ошибок. Это удобно показывать преподавателю без запуска ноутбука.

## 7. Синтетика: crop-ы объектов

```bash
python src/extract_classification_crops.py \
  --data-root data/coco10 \
  --out-root data/crops10 \
  --min-area 1024 \
  --max-per-class 700
```

## 8. Генерация через Stable Diffusion + ControlNet

Сначала выбери редкие классы по `reports/class_distribution.json`. Пример:

```bash
python src/generate_controlnet_synthetic.py \
  --input-root data/crops10/train \
  --output-root data/synth10 \
  --classes bicycle motorcycle truck \
  --num-per-class 80 \
  --steps 25
```

Важно: для ablation папка `data/synth10` должна иметь такие же подпапки классов, как `data/crops10/train`. Если синтетика сделана только для трёх классов, для остальных можно создать пустые папки или запускать классификационный ablation только на выбранном наборе классов.

## 9. Ablation: real vs real+synthetic

```bash
python src/train_classifier_ablation.py \
  --real-root data/crops10 \
  --synthetic-root data/synth10 \
  --output-dir runs/synthetic_ablation \
  --epochs 8
```

Результат: `runs/synthetic_ablation/ablation_metrics.csv`.

## 10. Быстрая проверка без COCO

Чтобы проверить, что структура проекта живая:

```bash
python src/make_toy_coco.py --out-root data/toy_coco
python src/pick_rare_classes.py --data-root data/toy_coco --out-json reports/toy_class_distribution.json
```

На toy-данных можно проверить чтение COCO-json, remap labels и генерацию служебных файлов. Полноценные метрики по DETR надо считать на настоящем COCO-subset.

## Что вставлять в итоговый отчёт

Минимальный набор:

- параметры запуска: классы, train/val size, epochs, lr, batch size;
- `val_metrics.csv` с mAP и mAP50;
- два графика loss;
- несколько изображений с предсказанными bbox;
- `summary.json` из error analysis;
- таблицу ablation для синтетики;
- короткий вывод: где модель ошибается и помогла ли синтетика.

## Мои ожидаемые наблюдения для отчёта

Числа надо брать только после реального запуска. Текстовые выводы можно писать так:

- DETR быстро начинает находить крупные и частые объекты, но на малом subset хуже работает с мелкими объектами и плотными сценами.
- `loss_bbox` и `loss_giou` полезнее смотреть вместе: bbox loss может снижаться, пока IoU всё ещё нестабилен.
- Ошибки локализации обычно появляются на частично закрытых объектах и объектах на границе кадра.
- Синтетика не обязана всегда улучшать качество. Если synthetic images слишком «чистые», модель может переобучиться на другой домен. Поэтому в отчёте важен ablation, а не только красивые картинки.
