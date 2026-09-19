"""
Run a Qwen vision-language model on every frame of every modality and save human
boxes, keypoints, body-part boxes and confidences in the same JSON layout as
detect_poses.py, so det_viewer.py can show them next to the YOLO results.

Layout:  <data_dir>/<episode>/<modality>/<frame>.png
Output:  { episode: { n: { modality: { "file": path,
                                       "<model>": [det, ...],              # verbal confidence
                                       "<model>-consensusN": [det, ...],   # only with --samples N
                                       "<model>_raw": "model reply" } } } }

Confidence is what the model says ("how likely is this a human"), not token log-probs.
With --samples N the model also answers N more times with sampling, and
conf = fraction of those answers that found the same person.

Usage:
    python qwen_classifier.py ../data
    python qwen_classifier.py ../data --model Qwen/Qwen3.5-4B --bits 16 --samples 5
"""
import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MAX_PIXELS = 1024 * 1024  # downscale big frames so image tokens fit in VRAM

# Same keypoints / parts as detect_poses.py
KPT_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
             "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
             "left_wrist", "right_wrist", "left_hip", "right_hip",
             "left_knee", "right_knee", "left_ankle", "right_ankle"]
PARTS = {
    "head": [0, 1, 2, 3, 4], "torso": [5, 6, 11, 12],
    "left_arm": [5, 7, 9], "right_arm": [6, 8, 10],
    "left_leg": [11, 13, 15], "right_leg": [12, 14, 16],
}

# One line telling the model what kind of image it is looking at
MOD_HINTS = {
    "rgb": "This is a normal color camera image.",
    "thermal": "This is a thermal infrared image; people usually appear bright.",
    "rf": "This is an RF (radio-frequency) sensor image rendered as a picture.",
}

PROMPT = """{hint}
This is an image representing a disaster scenario.
Find every human in the image, including small, far away, partly hidden or partly visible ones.
This could be represented as apendages (hands, feet, arms, legs, etc.) covered by some obstruction.
Include uncertain candidates too, with a low confidence.
For each human return {{"bbox_2d": [x1, y1, x2, y2], "confidence": 0.0-1.0 probability it is really a human,
"keypoints": {{name: [x, y, confidence] or null if not visible}}}} using these keypoint names:
{names}.
Coordinates are 0-1000, relative to image width and height.
False positives are discouraged, but false negatives are NEVER acceptable given the nature of search and rescue. 
Answer with only a JSON list. If there are no humans, answer []."""


def frame_number(path):
    """Pull the frame index n out of a filename, e.g. 'frame_0042.png' -> 42."""
    digits = re.findall(r"\d+", path.stem)
    return int(digits[-1]) if digits else path.stem


def part_boxes(kpts):
    """Box + mean confidence around each keypoint group; (0, 0) = not located."""
    parts = {}
    for name, idxs in PARTS.items():
        pts = [kpts[i] for i in idxs if kpts[i][0] > 0 or kpts[i][1] > 0]
        if pts:
            xs, ys, cs = zip(*pts)
            parts[name] = {"box": [min(xs), min(ys), max(xs), max(ys)],
                           "conf": round(sum(cs) / len(cs), 4)}
    return parts


def parse(text, w, h):
    """Model reply -> list of detections in pixel coordinates."""
    m = re.search(r"\[.*\]", text, re.S)  # the JSON list, ignoring ``` fences or chatter
    try:
        objs = json.loads(m.group()) if m else []
    except ValueError:
        return []
    dets = []
    for o in objs:
        try:  # skip any malformed object instead of the whole image
            x1, y1, x2, y2 = (float(v) for v in o["bbox_2d"])
            conf = float(o.get("confidence", 0.5))
            kpts = []
            for name in KPT_NAMES:
                p = (o.get("keypoints") or {}).get(name)
                c = float(p[2]) if p and len(p) > 2 else conf
                kpts.append([round(p[0] * w / 1000, 1), round(p[1] * h / 1000, 1), c] if p else [0, 0, 0])
            dets.append({
                "label": "person",
                "box": [round(x1 * w / 1000, 1), round(y1 * h / 1000, 1),
                        round(x2 * w / 1000, 1), round(y2 * h / 1000, 1)],
                "conf": conf,
                "keypoints": dict(zip(KPT_NAMES, kpts)),
                "parts": part_boxes(kpts),
            })
        except (TypeError, ValueError, KeyError, IndexError, AttributeError):
            continue
    return dets


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union > 0 else 0


def consensus(runs):
    """Match boxes across sampled answers (IoU > 0.5).
    conf = fraction of answers that found that person; the model's own score is kept as verbal_conf."""
    groups = []  # {"det": first det seen, "runs": answers that contained it}
    for r, dets in enumerate(runs):
        for d in dets:
            g = next((g for g in groups
                      if r not in g["runs"] and iou(g["det"]["box"], d["box"]) > 0.5), None)
            if g:
                g["runs"].add(r)
            else:
                groups.append({"det": d, "runs": {r}})
    return [{**g["det"], "verbal_conf": g["det"]["conf"], "conf": round(len(g["runs"]) / len(runs), 4)}
            for g in groups]


def ask(model, proc, img, prompt, sample=False):
    """Image + prompt -> reply text. Thinking off: grounding works better without it."""
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=2048, do_sample=sample,
                             **({"temperature": 0.8, "top_p": 0.95} if sample else {}))
    return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def main():
    ap = argparse.ArgumentParser(description="Qwen human detection -> viewer JSON")
    ap.add_argument("data_dir", type=Path, help="folder containing the episode folders")
    ap.add_argument("--out", type=Path, help="output JSON (default: <data_dir>/qwen_detections.json)")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B", help="Hugging Face model id")
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8, 16], help="weight precision")
    ap.add_argument("--samples", type=int, default=0, help="extra sampled answers for consensus conf (0 = off)")
    args = ap.parse_args()
    out_file = args.out or args.data_dir / "qwen_detections.json"

    # Load the model (quantized unless --bits 16; the vision encoder stays full precision)
    quant = None if args.bits == 16 else BitsAndBytesConfig(
        load_in_4bit=args.bits == 4, load_in_8bit=args.bits == 8,
        bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["visual", "lm_head"])
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", quantization_config=quant).eval()
    proc = AutoProcessor.from_pretrained(args.model, max_pixels=MAX_PIXELS)
    key = args.model.split("/")[-1].lower()  # e.g. "qwen3.5-9b" = source name in the viewer

    out = {}
    # Walk data/<episode>/<modality>/<frame>
    for ep in sorted(p for p in args.data_dir.iterdir() if p.is_dir()):
        for mod in sorted(p for p in ep.iterdir() if p.is_dir()):
            hint = MOD_HINTS.get(mod.name.lower(), f"This is a '{mod.name}' sensor image.")
            prompt = PROMPT.format(hint=hint, names=", ".join(KPT_NAMES))
            for img_path in sorted(mod.iterdir()):
                if img_path.suffix.lower() not in IMG_EXTS:
                    continue
                img = Image.open(img_path).convert("RGB")
                raw = ask(model, proc, img, prompt)
                entry = {"file": str(img_path), key: parse(raw, *img.size), f"{key}_raw": raw}
                if args.samples:
                    runs = [parse(ask(model, proc, img, prompt, sample=True), *img.size)
                            for _ in range(args.samples)]
                    entry[f"{key}-consensus{args.samples}"] = consensus(runs)
                # Group by episode -> frame n -> modality, like detect_poses.py
                out.setdefault(ep.name, {}).setdefault(str(frame_number(img_path)), {})[mod.name] = entry
                print(f"{ep.name}/{mod.name}/{img_path.name}: {len(entry[key])} humans")
            out_file.write_text(json.dumps(out))  # save per folder so a crash keeps finished work
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()