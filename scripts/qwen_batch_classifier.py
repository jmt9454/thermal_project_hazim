"""
Batched version of qwen_classifier.py for big GPUs (A100 etc.), using vLLM.
Same usage, prompt and output JSON, but many images are in flight at once so the
GPU stays busy instead of generating one reply at a time.

Layout:  <data_dir>/<episode>/<modality>/<frame>.png
Output:  { episode: { n: { modality: { "file": path,
                                       "<model>": [det, ...],              # verbal confidence
                                       "<model>-consensusN": [det, ...],   # only with --samples N
                                       "<model>_raw": "model reply" } } } }

Usage:
    python qwen_batch_classifier.py ../data
    python qwen_batch_classifier.py ../data --model Qwen/Qwen3.5-4B --bits 16 --samples 5

Needs vLLM (pip install vllm, Linux only). On an A100 --bits 16 is fastest;
4 and 8 bit only save memory.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# Same prompt, parsing and consensus as the one-at-a-time script
from qwen_classifier import (IMG_EXTS, KPT_NAMES, MAX_PIXELS, MOD_HINTS, PROMPT,
                             consensus, frame_number, parse)

CHUNK = 256          # images handed to vLLM per call; bounds RAM, vLLM batches within it
MAX_MODEL_LEN = 8192  # ~1k image tokens + prompt + 2048 reply; keeps KV cache small
QUANT = {4: "bitsandbytes", 8: "fp8", 16: None}  # fp8 runs weight-only on A100


def load(path):
    """-> (image downscaled for the model, original (w, h)).
    Replies use 0-1000 relative coords, so shrinking here doesn't change them."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = (MAX_PIXELS / (w * h)) ** 0.5
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    return img, (w, h)


def main():
    ap = argparse.ArgumentParser(description="Qwen human detection (batched, vLLM) -> viewer JSON")
    ap.add_argument("data_dir", type=Path, help="folder containing the episode folders")
    ap.add_argument("--out", type=Path, help="output JSON (default: <data_dir>/qwen_detections.json)")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B", help="Hugging Face model id")
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8, 16], help="weight precision")
    ap.add_argument("--samples", type=int, default=0, help="extra sampled answers for consensus conf (0 = off)")
    args = ap.parse_args()
    out_file = args.out or args.data_dir / "qwen_detections.json"

    llm = LLM(model=args.model, dtype="bfloat16", quantization=QUANT[args.bits],
              max_model_len=MAX_MODEL_LEN, limit_mm_per_prompt={"image": 1})
    proc = AutoProcessor.from_pretrained(args.model)
    key = args.model.split("/")[-1].lower()  # e.g. "qwen3.5-9b" = source name in the viewer

    greedy = SamplingParams(temperature=0, max_tokens=2048)
    # n answers per image share one prefill of the image
    sampled = SamplingParams(n=args.samples, temperature=0.8, top_p=0.95,
                             max_tokens=2048) if args.samples else None

    out = {}
    with ThreadPoolExecutor(8) as pool:  # decode PNGs in parallel while nothing else runs
        # Walk data/<episode>/<modality>/<frame>
        for ep in sorted(p for p in args.data_dir.iterdir() if p.is_dir()):
            for mod in sorted(p for p in ep.iterdir() if p.is_dir()):
                hint = MOD_HINTS.get(mod.name.lower(), f"This is a '{mod.name}' sensor image.")
                prompt = PROMPT.format(hint=hint, names=", ".join(KPT_NAMES))
                # Thinking off: grounding works better without it
                msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
                text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)

                paths = [p for p in sorted(mod.iterdir()) if p.suffix.lower() in IMG_EXTS]
                for i in range(0, len(paths), CHUNK):
                    chunk = paths[i:i + CHUNK]
                    imgs = list(pool.map(load, chunk))
                    reqs = [{"prompt": text, "multi_modal_data": {"image": img}} for img, _ in imgs]
                    replies = llm.generate(reqs, greedy, use_tqdm=False)
                    runs = llm.generate(reqs, sampled, use_tqdm=False) if sampled else [None] * len(reqs)

                    for img_path, (_, size), reply, run in zip(chunk, imgs, replies, runs):
                        raw = reply.outputs[0].text
                        entry = {"file": str(img_path), key: parse(raw, *size), f"{key}_raw": raw}
                        if run:
                            entry[f"{key}-consensus{args.samples}"] = consensus(
                                [parse(o.text, *size) for o in run.outputs])
                        # Group by episode -> frame n -> modality, like detect_poses.py
                        out.setdefault(ep.name, {}).setdefault(str(frame_number(img_path)), {})[mod.name] = entry
                        print(f"{ep.name}/{mod.name}/{img_path.name}: {len(entry[key])} humans")
                out_file.write_text(json.dumps(out))  # save per folder so a crash keeps finished work
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
