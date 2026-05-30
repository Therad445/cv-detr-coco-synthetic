# CV HW: DETR на COCO-subset + анализ ошибок

Учебный проект по computer vision: fine-tuning DETR на небольшом COCO-subset, логирование эксперимента, визуализация предсказаний и разбор ошибок. Вторая часть — дополнительный эксперимент с синтетикой для редких классов.

Я не реализую полный DETR с нуля. В работе используется предобученный `facebook/detr-resnet-50`, а основной фокус сделан на полном экспериментальном цикле: подготовка данных, обучение, метрики, TensorBoard, checkpoint, визуализация bbox и error analysis. Такая постановка ближе к финальному мини-проекту, чем к отдельному семинарскому упражнению.

## Связь с программой курса

- Базовая работа с изображениями, визуализация и сохранение артефактов — темы первых семинаров.
- ResNet/backbone используется внутри DETR как feature extractor.
- Detection-часть опирается на bbox, IoU, mAP/mAP50 и анализ ошибок локализации.
- DETR-часть связана с object queries, Hungarian matching и loss-компонентами `loss_ce`, `loss_bbox`, `loss_giou`.
- По стилю эксперимента работа похожа на fine-tuning-пайплайны из тем Mask2Former/Grounding DINO: Colab/GPU, subset данных, checkpoint, визуализации и reproducibility.

## Что сделано

1. Подготовка COCO-subset на 10 классов.
2. Fine-tuning DETR с логированием в TensorBoard.
3. Сохранение `train_losses.csv`, `val_metrics.csv`, checkpoint и profiler trace.
4. Графики loss-компонент.
5. Визуализация предсказанных bbox.
6. Error analysis: classification / localization / missed object / false positive.
7. Дополнительный блок синтетики: crop-ы объектов, генерация через ControlNet и ablation `real only` vs `real + synthetic`.

## Почему subset, а не полный COCO

Полный COCO тяжёлый для Colab и учебного fine-tuning. Я беру 10 классов:

```text
person, bicycle, car, motorcycle, bus, truck, cat, dog, chair, bottle
```

Так задача остаётся настоящей object detection, но её можно прогнать на одной T4. После подготовки subset я отдельно смотрю распределение классов: частые классы вроде `person` доминируют, а `cat`, `dog`, `motorcycle` обычно оказываются редкими. Это нужно для честного анализа ошибок и для synthetic-эксперимента.

## Сколько эпох запускать

Есть три режима:

| Режим | Эпохи | Для чего |
|---|---:|---|
| Debug | 1 | Проверить, что pipeline работает: loss пишется, checkpoint создаётся. |
| Normal Colab | 5–10 | Получить первые осмысленные графики и визуализации. |
| Strong run | 20–30 | Лучше для финальной сдачи, если хватает GPU-времени. |

Важно: для предобученного DETR fine-tuning на маленьком subset 5 эпох могут быть достаточны как учебный запуск, но не как сильный финальный эксперимент. Если есть время, лучше запускать 20–30 эпох и смотреть не только число эпох, а динамику `val mAP/mAP50`. Если mAP перестала расти, дальше обучение не обязательно полезно.

В Colab лучше сразу писать результаты в Google Drive:

```bash
--output-dir /content/drive/MyDrive/CV_HW_DETR_RUNS/detr_coco10
```

Иначе runtime может умереть и стереть `/content`.

## Установка

```bash
pip install -r requirements.txt
```

Проверка проекта:

```bash
python -m compileall -q src
python src/make_toy_coco.py --out-root data/toy_coco
python src/pick_rare_classes.py --data-root data/toy_coco --out-json reports/toy_class_distribution.json
```

## Подготовка COCO-subset

Ожидаемая структура COCO:

```text
data/coco/
  train2017/
  val2017/
  annotations/
    instances_train2017.json
    instances_val2017.json
```

Команда:

```bash
python src/prepare_coco_subset.py \
  --coco-root data/coco \
  --out-root data/coco10 \
  --classes person bicycle car motorcycle bus truck cat dog chair bottle \
  --max-train-images 1200 \
  --max-val-images 300 \
  --link-mode copy

python src/pick_rare_classes.py \
  --data-root data/coco10 \
  --out-json reports/class_distribution.json
```

## Обучение DETR

Debug-запуск:

```bash
python src/train_detr.py \
  --data-root data/coco10 \
  --model-name facebook/detr-resnet-50 \
  --output-dir runs/detr_coco10_debug \
  --epochs 1 \
  --batch-size 2 \
  --lr 1e-5 \
  --lr-backbone 1e-6 \
  --weight-decay 1e-4 \
  --num-workers 2 \
  --profile
```

Финальный Colab-запуск в Drive:

```bash
python src/train_detr.py \
  --data-root data/coco10 \
  --model-name facebook/detr-resnet-50 \
  --output-dir /content/drive/MyDrive/CV_HW_DETR_RUNS/detr_coco10 \
  --epochs 20 \
  --batch-size 2 \
  --lr 1e-5 \
  --lr-backbone 1e-6 \
  --weight-decay 1e-4 \
  --num-workers 2 \
  --profile
```

Если запуск оборвался, можно продолжить с checkpoint:

```bash
python src/train_detr.py \
  --data-root data/coco10 \
  --output-dir /content/drive/MyDrive/CV_HW_DETR_RUNS/detr_coco10 \
  --epochs 20 \
  --resume /content/drive/MyDrive/CV_HW_DETR_RUNS/detr_coco10/checkpoints/epoch_005.pt
```

## Графики, визуализации и отчёт

```bash
python src/plot_losses.py \
  --log-csv runs/detr_coco10/train_losses.csv \
  --out-dir reports/figures

python src/visualize_predictions.py \
  --data-root data/coco10 \
  --checkpoint runs/detr_coco10/checkpoints/best.pt \
  --out-dir reports/figures/predictions \
  --num-images 24 \
  --score-threshold 0.3

python src/error_analysis.py \
  --data-root data/coco10 \
  --checkpoint runs/detr_coco10/checkpoints/best.pt \
  --out-dir reports/error_analysis \
  --score-threshold 0.3 \
  --iou-threshold 0.5

python src/summarize_run.py \
  --run-dir runs/detr_coco10 \
  --reports-dir reports \
  --out-md reports/run_summary.md

python src/make_demo_report.py \
  --run-dir runs/detr_coco10 \
  --reports-dir reports \
  --out-file reports/demo_report.html
```

## Синтетика и ablation

Сначала извлекаются crop-ы объектов:

```bash
python src/extract_classification_crops.py \
  --data-root data/coco10 \
  --out-root data/crops10 \
  --min-area 1024 \
  --max-per-class 700
```

Затем можно генерировать синтетику для редких классов:

```bash
python src/generate_controlnet_synthetic.py \
  --input-root data/crops10/train \
  --output-root data/synth10 \
  --classes cat motorcycle dog \
  --num-per-class 40 \
  --steps 25
```

Ablation:

```bash
python src/train_classifier_ablation.py \
  --real-root data/crops10 \
  --synthetic-root data/synth10 \
  --output-dir runs/synthetic_ablation \
  --epochs 8
```

Синтетика не обязана улучшать качество. Из-за domain gap она может ухудшить результат, поэтому здесь важен именно ablation, а не красивые картинки.

## Что показывать на защите

1. Структуру проекта.
2. Smoke-test на toy COCO.
3. Таблицу классов COCO-subset и дисбаланс.
4. Параметры обучения DETR.
5. TensorBoard / loss-графики.
6. `val_metrics.csv` с `mAP/mAP50`.
7. Примеры bbox-предсказаний.
8. Error analysis.
9. Synthetic-блок как дополнительный эксперимент.

Главная идея работы: не просто запустить DETR, а собрать полный цикл CV-эксперимента — данные, обучение, метрики, визуализации, анализ ошибок и проверку гипотезы про синтетику.
