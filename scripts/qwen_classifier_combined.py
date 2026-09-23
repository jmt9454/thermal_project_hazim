"""
Like qwen_classifier.py, but the RGB and thermal image of each frame go to the model
together in one message, and it answers once for the pair.

Layout:  <data_dir>/<episode>/<modality>/<frame>.png, with an RGB folder ("rgb") and a
         thermal folder ("ir" or "thermal", any case) in each episode. Images are paired
         by frame number; a frame missing either image is skipped.
Output:  { episode: { n: { modality: { "file": path,
                                       "<model>-combined": [det, ...],              # verbal confidence
                                       "<model>-combined-consensusN": [det, ...],   # only with --samples N
                                       "<model>-combined_raw": "model reply" } } } }
The pair's one reply is stored under both modalities, in each image's own pixels, so
results_db.py, evaluate.py and detection_viewer.py treat "<model>-combined" like any
other model (its RGB and thermal rows are the same answer).

The prompt is qwen_classifier.py's with the one-line modality hint swapped for a
description of the pair, so results compare directly with the one-image runs.
The two images must be pixel-aligned (same camera pose), as in the simulator episodes.

Usage:
    python qwen_classifier_combined.py ../data
    python qwen_classifier_combined.py ../data --model Qwen/Qwen3.5-4B --bits 16 --samples 5
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

# Same prompt, parsing and consensus as the one-image script
from qwen_classifier import (IMG_EXTS, KPT_NAMES, MAX_PIXELS, PROMPT, consensus,
                             frame_number, parse)

# Modality folder names (any case) that make up a pair
RGB_DIRS = {"rgb"}
THERMAL_DIRS = {"ir", "thermal"}

# Takes the place of the one-line modality hint in PROMPT; the images are sent in this order
HINT = ("You are given two images of the same scene, taken from the same viewpoint at the same moment: "
        "first an RGB image, then a thermal/IR image in which people usually appear bright. "
        "They are pixel-aligned, so treat them as one image: a person seen in either counts, "
        "and a box or keypoint marks the same place in both.")


def find_pairs(ep):
    """Episode folder -> ((rgb modality, thermal modality), [(rgb path, thermal path), ...])
    for every frame number found in both folders."""
    rgb_dir, th_dir = (next((p for p in sorted(ep.iterdir()) if p.is_dir() and p.name.lower() in names), None)
                       for names in (RGB_DIRS, THERMAL_DIRS))
    if not (rgb_dir and th_dir):
        print(f"skipped {ep.name}: needs an RGB and a thermal folder")
        return (), []
    rgb, th = ({frame_number(p): p for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXTS}
               for d in (rgb_dir, th_dir))
    if rgb.keys() != th.keys():
        print(f"{ep.name}: skipped {len(rgb.keys() ^ th.keys())} frames that have only one of the two images")
    return (rgb_dir.name, th_dir.name), [(p, th[n]) for n, p in rgb.items() if n in th]


def ask(model, proc, imgs, prompt, sample=False):
    """Images + prompt in one message -> reply text. Thinking off: grounding works better without it."""
    msgs = [{"role": "user", "content": [{"type": "image"} for _ in imgs] + [{"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = proc(text=[text], images=imgs, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=2048, do_sample=sample,
                             **({"temperature": 0.8, "top_p": 0.95} if sample else {}))
    return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def main():
    ap = argparse.ArgumentParser(description="Qwen human detection on RGB + thermal pairs -> viewer JSON")
    ap.add_argument("data_dir", type=Path, help="folder containing the episode folders")
    ap.add_argument("--out", type=Path, help="output JSON (default: <data_dir>/qwen_combined_detections.json)")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B", help="Hugging Face model id")
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8, 16], help="weight precision")
    ap.add_argument("--samples", type=int, default=0, help="extra sampled answers for consensus conf (0 = off)")
    args = ap.parse_args()
    out_file = args.out or args.data_dir / "qwen_combined_detections.json"

    # Load the model (quantized unless --bits 16; the vision encoder stays full precision)
    quant = None if args.bits == 16 else BitsAndBytesConfig(
        load_in_4bit=args.bits == 4, load_in_8bit=args.bits == 8,
        bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["visual", "lm_head"])
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", quantization_config=quant).eval()
    proc = AutoProcessor.from_pretrained(args.model, max_pixels=MAX_PIXELS)
    key = args.model.split("/")[-1].lower() + "-combined"  # e.g. "qwen3.5-9b-combined" = source name in the viewer
    prompt = PROMPT.format(hint=HINT, names=", ".join(KPT_NAMES))

    out = {}
    # Walk data/<episode>/<modality>/<frame>, one RGB + thermal pair per frame
    for ep in sorted(p for p in args.data_dir.iterdir() if p.is_dir()):
        mods, pairs = find_pairs(ep)
        for paths in pairs:
            imgs = [Image.open(p).convert("RGB") for p in paths]
            raw = ask(model, proc, imgs, prompt)
            samples = [ask(model, proc, imgs, prompt, sample=True) for _ in range(args.samples)]
            # One answer for the pair, stored under both modalities in each image's own pixels
            for mod, img_path, img in zip(mods, paths, imgs):
                entry = {"file": str(img_path), key: parse(raw, *img.size), f"{key}_raw": raw}
                if args.samples:
                    entry[f"{key}-consensus{args.samples}"] = consensus([parse(s, *img.size) for s in samples])
                # Group by episode -> frame n -> modality, like detect_poses.py
                out.setdefault(ep.name, {}).setdefault(str(frame_number(img_path)), {})[mod] = entry
            print(f"{ep.name}/{'+'.join(mods)}/{paths[0].name}: {len(entry[key])} humans")
        out_file.write_text(json.dumps(out))  # save per folder so a crash keeps finished work
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
