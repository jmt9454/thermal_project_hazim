"""
Batched version of qwen_classifier_combined_rf.py for big GPUs (A100 etc.), using vLLM.
Same usage, prompts and output JSON, but many image pairs are in flight at once so the
GPU stays busy instead of generating one reply at a time.

Layout:  <data_dir>/<episode>/<modality>/<frame>.png, with an RGB and a thermal folder
         in each episode and the RF readings in the <frame>.json next to each RGB image
         (see qwen_classifier_combined_rf.py)
Output:  { episode: { n: { modality: { "file": path,
                                       "<model>-combined-rf": [det, ...],              # verbal confidence
                                       "<model>-combined-rf-consensusN": [det, ...],   # only with --samples N
                                       "<model>-combined-rf_raw": "model reply",
                                       "rf_phone": rf_sources entry or null } } } }    # the phone in the prompt
         the pair's one reply is stored under both modalities

Usage:
    python qwen_classifier_combined_rf_batch.py ../data
    python qwen_classifier_combined_rf_batch.py ../data --model Qwen/Qwen3.5-4B --bits 16 --samples 5

Needs vLLM (pip install vllm, Linux only). On an A100 --bits 16 is fastest;
4 and 8 bit only save memory.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# Same image loading and engine settings as the one-image batch script, same pairing
# as the RGB+IR script, same prompts as the one-at-a-time RF script
from qwen_batch_classifier import CHUNK, MAX_MODEL_LEN, QUANT, load
from qwen_classifier import consensus, frame_number, parse
from qwen_classifier_combined import find_pairs
from qwen_classifier_combined_rf import closest_phone, make_prompt, rf_note


def load_pair(paths):
    """(rgb path, thermal path) -> [(image, (w, h)), ...] from load(), or None if either is unreadable."""
    pair = [load(p) for p in paths]
    return pair if all(pair) else None


def chat(proc, prompt):
    """Prompt -> chat text for one pair: color image, thermal image, instructions.
    Thinking off: grounding works better without it."""
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": prompt}]}]
    return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def main():
    ap = argparse.ArgumentParser(
        description="Qwen human detection on RGB + thermal pairs with RF hints (batched, vLLM) -> viewer JSON")
    ap.add_argument("data_dir", type=Path, help="folder containing the episode folders")
    ap.add_argument("--out", type=Path, help="output JSON (default: <data_dir>/qwen_combined_rf_detections.json)")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B", help="Hugging Face model id")
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8, 16], help="weight precision")
    ap.add_argument("--samples", type=int, default=0, help="extra sampled answers for consensus conf (0 = off)")
    args = ap.parse_args()
    out_file = args.out or args.data_dir / "qwen_combined_rf_detections.json"

    llm = LLM(model=args.model, dtype="bfloat16", quantization=QUANT[args.bits],
              max_model_len=MAX_MODEL_LEN, limit_mm_per_prompt={"image": 2, "video": 0})
    proc = AutoProcessor.from_pretrained(args.model)
    # e.g. "qwen3.5-9b-combined-rf" = source name in the viewer, kept apart from the RGB+IR run's "-combined"
    key = args.model.split("/")[-1].lower() + "-combined-rf"

    greedy = SamplingParams(temperature=0, max_tokens=2048)
    # n answers per pair share one prefill of the images
    sampled = SamplingParams(n=args.samples, temperature=0.8, top_p=0.95,
                             max_tokens=2048) if args.samples else None

    out = {}
    with ThreadPoolExecutor(8) as pool:  # decode PNGs in parallel while nothing else runs
        # Walk data/<episode>/<modality>/<frame>, one RGB + thermal pair per frame
        for ep in sorted(p for p in args.data_dir.iterdir() if p.is_dir()):
            mods, pairs = find_pairs(ep)
            for i in range(0, len(pairs), CHUNK):
                loaded = [(p, r) for p, r in zip(pairs[i:i + CHUNK], pool.map(load_pair, pairs[i:i + CHUNK])) if r]
                if not loaded:
                    continue
                chunk, imgs = zip(*loaded)
                # A prompt per pair, since the RF sentence depends on the frame's nearest phone
                phones = [closest_phone(rgb) for rgb, _ in chunk]
                reqs = [{"prompt": chat(proc, make_prompt(phone)),
                         "multi_modal_data": {"image": [img for img, _ in pair]}}
                        for phone, pair in zip(phones, imgs)]
                replies = llm.generate(reqs, greedy, use_tqdm=False)
                runs = llm.generate(reqs, sampled, use_tqdm=False) if sampled else [None] * len(reqs)

                for paths, pair, phone, reply, run in zip(chunk, imgs, phones, replies, runs):
                    raw = reply.outputs[0].text
                    # One answer for the pair, stored under both modalities in each image's own pixels
                    for mod, img_path, (_, size) in zip(mods, paths, pair):
                        entry = {"file": str(img_path), key: parse(raw, *size), f"{key}_raw": raw,
                                 "rf_phone": phone}
                        if run:
                            entry[f"{key}-consensus{args.samples}"] = consensus(
                                [parse(o.text, *size) for o in run.outputs])
                        # Group by episode -> frame n -> modality, like detect_poses.py
                        out.setdefault(ep.name, {}).setdefault(str(frame_number(img_path)), {})[mod] = entry
                    print(f"{ep.name}/{'+'.join(mods)}/{paths[0].name}: {len(entry[key])} humans{rf_note(phone)}")
            out_file.write_text(json.dumps(out))  # save per folder so a crash keeps finished work
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
