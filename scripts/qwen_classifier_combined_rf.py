"""
Like qwen_classifier_combined.py (the RGB and thermal image of each frame go to the model
together in one message), but when the RF sensor picks up a phone in view, the prompt also
gives the nearest one's signal strength and distance.

This is the RGB+IR+RF run, compared with the one-image runs (qwen_classifier.py) and the
RGB+IR run (qwen_classifier_combined.py). It reuses that script's pairing, hint and model
call, and a frame with no phone in view gets its prompt unchanged, so the RF sentence is
the only difference between the two runs.

Layout:  <data_dir>/<episode>/<modality>/<frame>.png, with an RGB folder ("rgb") and a
         thermal folder ("ir" or "thermal", any case) in each episode. The RF readings are
         the "rf_sources" list in the <frame>.json next to each RGB image (in_frame is
         added by rf_mod.py).
Output:  { episode: { n: { modality: { "file": path,
                                       "<model>-combined-rf": [det, ...],              # verbal confidence
                                       "<model>-combined-rf-consensusN": [det, ...],   # only with --samples N
                                       "<model>-combined-rf_raw": "model reply",
                                       "rf_phone": rf_sources entry or null } } } }    # the phone in the prompt
The pair's one reply is stored under both modalities, as in qwen_classifier_combined.py.
"rf_phone" is not a list, so results_db.py and detection_viewer.py skip it like "file".

Usage:
    python qwen_classifier_combined_rf.py ../data
    python qwen_classifier_combined_rf.py ../data --model Qwen/Qwen3.5-4B --bits 16 --samples 5
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

# Same prompt, parsing and consensus as the one-image script,
# same pairing, hint and model call as the RGB+IR script
from qwen_classifier import KPT_NAMES, MAX_PIXELS, PROMPT, consensus, frame_number, parse
from qwen_classifier_combined import HINT, ask, find_pairs

# Added after HINT when a phone is in view
RF_HINT = (" Radio frequency from a cell phone within the field of view has been detected at {dbm:.0f} dBm, "
           "at an estimated distance of {dist:.0f} meters.")


def closest_phone(rgb_path):
    """RGB image -> the nearest phone in its frame JSON that is in frame and detected, or None.
    A missing or unreadable JSON counts as no phone."""
    try:
        # bytes, not text: json detects the encoding (a few frame JSONs are UTF-16, see evaluate.py)
        sources = json.loads(rgb_path.with_suffix(".json").read_bytes()).get("rf_sources") or []
    except (OSError, ValueError):
        return None
    phones = [s for s in sources  # "type" is written both "Phone" and "phone"
              if s.get("type", "").lower() == "phone" and s.get("in_frame") and s.get("detected")]
    return min(phones, key=lambda s: s["distance_m"], default=None)


def make_prompt(phone):
    """Phone from closest_phone() -> full prompt; None gives the RGB+IR prompt unchanged."""
    rf = RF_HINT.format(dbm=phone["strength_dbm"], dist=phone["distance_m"]) if phone else ""
    return PROMPT.format(hint=HINT + rf, names=", ".join(KPT_NAMES))


def rf_note(phone):
    """Log suffix for the phone in the prompt, so frames that got one are easy to spot."""
    return f", RF phone at {phone['distance_m']:.0f} m, {phone['strength_dbm']:.0f} dBm" if phone else ""


def main():
    ap = argparse.ArgumentParser(description="Qwen human detection on RGB + thermal pairs with RF hints -> viewer JSON")
    ap.add_argument("data_dir", type=Path, help="folder containing the episode folders")
    ap.add_argument("--out", type=Path, help="output JSON (default: <data_dir>/qwen_combined_rf_detections.json)")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B", help="Hugging Face model id")
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8, 16], help="weight precision")
    ap.add_argument("--samples", type=int, default=0, help="extra sampled answers for consensus conf (0 = off)")
    args = ap.parse_args()
    out_file = args.out or args.data_dir / "qwen_combined_rf_detections.json"

    # Load the model (quantized unless --bits 16; the vision encoder stays full precision)
    quant = None if args.bits == 16 else BitsAndBytesConfig(
        load_in_4bit=args.bits == 4, load_in_8bit=args.bits == 8,
        bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["visual", "lm_head"])
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", quantization_config=quant).eval()
    proc = AutoProcessor.from_pretrained(args.model, max_pixels=MAX_PIXELS)
    # e.g. "qwen3.5-9b-combined-rf" = source name in the viewer. Not "-combined": results_db.py
    # keeps one result per image and model name, so it would replace the RGB+IR run's
    key = args.model.split("/")[-1].lower() + "-combined-rf"

    out = {}
    # Walk data/<episode>/<modality>/<frame>, one RGB + thermal pair per frame
    for ep in sorted(p for p in args.data_dir.iterdir() if p.is_dir()):
        mods, pairs = find_pairs(ep)
        for paths in pairs:
            imgs = [Image.open(p).convert("RGB") for p in paths]
            phone = closest_phone(paths[0])  # the RF readings are in the RGB image's JSON
            prompt = make_prompt(phone)
            raw = ask(model, proc, imgs, prompt)
            samples = [ask(model, proc, imgs, prompt, sample=True) for _ in range(args.samples)]
            # One answer for the pair, stored under both modalities in each image's own pixels
            for mod, img_path, img in zip(mods, paths, imgs):
                entry = {"file": str(img_path), key: parse(raw, *img.size), f"{key}_raw": raw, "rf_phone": phone}
                if args.samples:
                    entry[f"{key}-consensus{args.samples}"] = consensus([parse(s, *img.size) for s in samples])
                # Group by episode -> frame n -> modality, like detect_poses.py
                out.setdefault(ep.name, {}).setdefault(str(frame_number(img_path)), {})[mod] = entry
            print(f"{ep.name}/{'+'.join(mods)}/{paths[0].name}: {len(entry[key])} humans{rf_note(phone)}")
        out_file.write_text(json.dumps(out))  # save per folder so a crash keeps finished work
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
