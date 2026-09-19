"""
Run YOLOv8-pose and YOLO26-pose on every frame of every modality in data/
and save ALL human detections (person boxes, 17 keypoints, body-part boxes)
to one JSON file. No confidence threshold is applied - filter in the viewer.

Usage:
    python yolo_classifier_8_26.py /path/to/data [--out detections.json]

Expected layout:
    <data_dir>/<episode>/<modality>/<frame>.png   e.g. data/ep1/thermal/0042.png
The number in the frame filename (n) links the same frame across modalities.

Output (<data_dir>/detections.json unless --out is given):
    { episode: { n: { modality: { "file": path,
                                  "yolov8": [det, ...],
                                  "yolo26": [det, ...] } } } }
    det = { "label": "person", "box": [x1,y1,x2,y2], "conf": c,
            "keypoints": { "nose": [x,y,c], ... },
            "parts":     { "head": {"box": [...], "conf": c}, ... } }
"""
import argparse
import json
import re
from pathlib import Path

from ultralytics import YOLO

# Weights download automatically on first use. Swap 'x' for n/s/m/l for speed.
MODELS = {"yolov8": "yolov8x-pose.pt", "yolo26": "yolo26x-pose.pt"}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# COCO 17-keypoint names, in the order YOLO pose models output them
KPT_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
             "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
             "left_wrist", "right_wrist", "left_hip", "right_hip",
             "left_knee", "right_knee", "left_ankle", "right_ankle"]

# Body parts = a box drawn around a group of keypoints (indices into KPT_NAMES)
PARTS = {
    "head":      [0, 1, 2, 3, 4],
    "torso":     [5, 6, 11, 12],
    "left_arm":  [5, 7, 9],
    "right_arm": [6, 8, 10],
    "left_leg":  [11, 13, 15],
    "right_leg": [12, 14, 16],
}


def frame_number(path):
    """Pull the frame index n out of a filename, e.g. 'frame_0042.png' -> 42."""
    digits = re.findall(r"\d+", path.stem)
    return int(digits[-1]) if digits else path.stem


def r(values, nd=2):
    """Round a list of floats to keep the JSON small."""
    return [round(v, nd) for v in values]


def part_boxes(kpts):
    """Build a box + confidence for each body part from its keypoints.
    kpts: list of [x, y, conf]. Keypoints at (0, 0) were not located, so skip them.
    Part confidence = mean confidence of the keypoints that make it up."""
    parts = {}
    for name, idxs in PARTS.items():
        pts = [kpts[i] for i in idxs if kpts[i][0] > 0 or kpts[i][1] > 0]
        if not pts:
            continue
        xs, ys, cs = zip(*pts)
        parts[name] = {"box": r([min(xs), min(ys), max(xs), max(ys)]),
                       "conf": round(sum(cs) / len(cs), 4)}
    return parts


def detect(model, img_path):
    """Run one pose model on one image and return a list of detections."""
    res = model.predict(str(img_path), conf=0.0, verbose=False)[0]  # conf=0 -> keep everything
    boxes = res.boxes
    kpts = res.keypoints.data.tolist() if res.keypoints is not None else [None] * len(boxes)

    dets = []
    for box, conf, cls, kp in zip(boxes.xyxy.tolist(), boxes.conf.tolist(),
                                  boxes.cls.tolist(), kpts):
        det = {"label": res.names[int(cls)],   # always "person" for COCO pose models
               "box": r(box),
               "conf": round(conf, 4)}
        if kp:
            det["keypoints"] = {n: r(p, 4) for n, p in zip(KPT_NAMES, kp)}
            det["parts"] = part_boxes(kp)
        dets.append(det)
    return dets


def main():
    # Data dir (parent of ep1, ep2, ...) comes from the command line
    ap = argparse.ArgumentParser(description="YOLO pose detections -> JSON")
    ap.add_argument("data_dir", type=Path, help="folder containing the episode folders")
    ap.add_argument("--out", type=Path, help="output JSON (default: <data_dir>/detections.json)")
    args = ap.parse_args()
    out_file = args.out or args.data_dir / "detections.json"

    models = {name: YOLO(weights) for name, weights in MODELS.items()}
    out = {}

    # Walk data/<episode>/<modality>/<frame>
    for ep in sorted(p for p in args.data_dir.iterdir() if p.is_dir()):
        for mod in sorted(p for p in ep.iterdir() if p.is_dir()):
            for img in sorted(mod.iterdir()):
                if img.suffix.lower() not in IMG_EXTS:
                    continue
                n = str(frame_number(img))
                # Group by episode -> frame n -> modality, so modalities of a frame sit together
                out.setdefault(ep.name, {}).setdefault(n, {})[mod.name] = {
                    "file": str(img),
                    **{name: detect(model, img) for name, model in models.items()},
                }
            print(f"done: {ep.name}/{mod.name}")

    out_file.write_text(json.dumps(out))
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()