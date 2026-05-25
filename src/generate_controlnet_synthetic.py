import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from utils import mkdir, set_seed


def canny_image(img, low=100, high=200):
    arr = np.array(img.convert('RGB'))
    edges = cv2.Canny(arr, low, high)
    edges = np.stack([edges, edges, edges], axis=2)
    return Image.fromarray(edges)


def load_pipe(device):
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

    dtype = torch.float16 if device.startswith('cuda') else torch.float32
    controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-canny', torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    if device.startswith('cuda'):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    return pipe


def prompt_for_class(cls):
    return (
        f'realistic photo of a {cls}, ordinary background, natural light, '
        'dataset image, object is visible, sharp focus, not a studio render'
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-root', type=Path, required=True, help='data/crops10/train')
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--classes', nargs='+', required=True)
    p.add_argument('--num-per-class', type=int, default=80)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--steps', type=int, default=25)
    p.add_argument('--guidance-scale', type=float, default=7.5)
    args = p.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    mkdir(args.output_root)
    pipe = load_pipe(args.device)

    meta_file = args.output_root / 'metadata.csv'
    exists = meta_file.exists()
    with open(meta_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['class_name', 'output_file', 'source_file', 'prompt', 'seed'])
        if not exists:
            writer.writeheader()

        for cls in args.classes:
            src_dir = args.input_root / cls
            src_imgs = sorted([p for p in src_dir.glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}])
            if not src_imgs:
                print('skip, no sources:', cls)
                continue
            out_dir = args.output_root / cls
            mkdir(out_dir)
            prompt = prompt_for_class(cls)
            negative = 'cartoon, anime, painting, text, watermark, duplicate object, deformed, low quality'

            for i in tqdm(range(args.num_per_class), desc='synth ' + cls):
                src = random.choice(src_imgs)
                base = Image.open(src).convert('RGB').resize((512, 512))
                control = canny_image(base)
                gen = torch.Generator(device=args.device).manual_seed(args.seed + 1000 * args.classes.index(cls) + i)
                img = pipe(
                    prompt=prompt,
                    negative_prompt=negative,
                    image=control,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    generator=gen,
                ).images[0]
                out_path = out_dir / f'{cls}_synth_{i:04d}.png'
                img.save(out_path)
                writer.writerow({
                    'class_name': cls,
                    'output_file': str(out_path),
                    'source_file': str(src),
                    'prompt': prompt,
                    'seed': args.seed + i,
                })

    print('saved synthetic images:', args.output_root)


if __name__ == '__main__':
    main()
