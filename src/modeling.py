from pathlib import Path

import torch
from transformers import DetrForObjectDetection, DetrImageProcessor

from utils import read_json


def load_labels(data_root):
    data = read_json(Path(data_root) / 'annotations' / 'id2label.json')
    id2label = {int(k): v for k, v in data['id2label'].items()}
    label2id = {k: int(v) for k, v in data['label2id'].items()}
    return id2label, label2id


def get_processor(model_name):
    return DetrImageProcessor.from_pretrained(model_name)


def get_model(model_name, data_root):
    id2label, label2id = load_labels(data_root)
    model = DetrForObjectDetection.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model, id2label, label2id


def load_model_from_checkpoint(ckpt_path, data_root, model_name=None):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    saved_args = ckpt.get('args', {})
    model_name = model_name or saved_args.get('model_name') or 'facebook/detr-resnet-50'
    model, id2label, label2id = get_model(model_name, data_root)
    model.load_state_dict(ckpt.get('model_state', ckpt), strict=True)
    return model, id2label, label2id
