"""
Score YOLO / Qwen detections against the per-frame ground truth JSONs.

Ground truth: <data_dir>/<episode>/<traj>/<modality>/frame_<n>.json, one point per
target (screen_x, screen_y in 0-1) plus in_frame / visible / occluded flags.
Detections: a detections.db built from the classifier JSONs by results_db.py.

Matching: a detection hits a target when the target's point lies inside its box
(grown by --margin px). Detections are taken highest confidence first and each
claims at most one unclaimed target (nearest to the box centre), so a duplicate
box on the same person is a false positive. Animal targets (pigs, dogs) are never
positives: a box on one counts as a false positive and is reported as animal_hits.
Out-of-frame targets are ignored.

Usage:
    python results_db.py ../results                  # once, after new detections
    python evaluate.py ../data ../results/detections.db
    python evaluate.py ../data ../results/detections.db --conf 0.3 --margin 10

detection_viewer.py uses match() / load_targets() from here, so the frame view
colours hits and misses exactly as they are scored.

Writes to <database folder>/eval/:
    summary.csv  one row per model x modality at --conf
    curves.csv   precision / recall / FP-per-image at thresholds 0.00-1.00
    targets.csv  every in-frame target x model x modality with the confidence of
                 the detection that hit it (empty = missed) - for custom plots
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from results_db import open_db, unpack_boxes

ANIMALS = ("pig", "dog")  # target names containing these, in any case, are not people (pig1, dog2, SK_Pig_Skeleton)
THRESHOLDS = np.round(np.arange(0, 1.0001, 0.05), 2)


def frame_number(path):
    """Pull the frame index n out of a filename, e.g. 'frame_0042.png' -> 42."""
    digits = re.findall(r"\d+", path.stem)
    return int(digits[-1]) if digits else path.stem


def index_ground_truth(data_dir):
    """-> {(episode, traj, modality, frame): gt json path}, {(episode, traj, modality): (w, h)}"""
    gt, sizes = {}, {}
    for f in data_dir.glob("*/*/*/*.json"):
        mod_dir = f.parent
        key = (mod_dir.parent.parent.name, mod_dir.parent.name, mod_dir.name)
        gt[(*key, frame_number(f))] = f
        if key not in sizes:
            img = next((p for p in mod_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")), None)
            sizes[key] = Image.open(img).size if img else (1920, 1080)
    return gt, sizes


def load_targets(path, w, h):
    """In-frame targets -> (names, xy pixels [k,2], is_human [k], visible [k])."""
    names, xy, human, visible = [], [], [], []
    # bytes, not text: json detects the encoding, and a few GT files are UTF-16
    for t in json.loads(path.read_bytes())["targets"]:
        if not t["in_frame"] or not (0 <= t["screen_x"] <= 1 and 0 <= t["screen_y"] <= 1):
            continue
        names.append(t["name"])
        xy.append((t["screen_x"] * w, t["screen_y"] * h))
        human.append(not any(a in t["name"].lower() for a in ANIMALS))
        visible.append(bool(t["visible"]))
    return names, np.array(xy).reshape(-1, 2), np.array(human, bool), np.array(visible, bool)


def match(conf, boxes, xy, margin):
    """Greedy one-to-one matching of person detections to target points, highest
    confidence first.  conf [d], boxes [d, 4], xy [k, 2]
    -> hit target index per detection, in the input order (-1 = none)"""
    hit = np.full(len(conf), -1)
    if not len(conf) or not len(xy):
        return hit
    order = np.argsort(-conf, kind="stable")  # ties keep file order
    boxes = boxes[order]
    x1, y1 = boxes[:, :1] - margin, boxes[:, 1:2] - margin
    x2, y2 = boxes[:, 2:3] + margin, boxes[:, 3:4] + margin
    inside = (xy[:, 0] >= x1) & (xy[:, 0] <= x2) & (xy[:, 1] >= y1) & (xy[:, 1] <= y2)  # [d, k]
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    taken = np.zeros(len(xy), bool)
    for i in np.flatnonzero(inside.any(1)):
        cand = np.flatnonzero(inside[i] & ~taken)
        if len(cand):
            j = cand[np.argmin(np.hypot(xy[cand, 0] - cx[i], xy[cand, 1] - cy[i]))]
            hit[order[i]], taken[j] = j, True
    return hit


def average_precision(conf, tp, n_pos):
    """All-point interpolated AP (area under the precision envelope)."""
    if n_pos == 0 or len(conf) == 0:
        return float("nan")
    order = np.argsort(-conf, kind="stable")
    ctp = np.cumsum(tp[order])
    recall = ctp / n_pos
    precision = ctp / np.arange(1, len(ctp) + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(np.concatenate([[0], recall])) * precision))


def main():
    ap = argparse.ArgumentParser(description="Detections vs ground truth -> precision / recall tables")
    ap.add_argument("data_dir", type=Path, help="folder containing the episode folders (with GT jsons)")
    ap.add_argument("db", type=Path, help="detections.db from results_db.py")
    ap.add_argument("--conf", type=float, default=0.5, help="threshold for summary.csv (default 0.5)")
    ap.add_argument("--margin", type=float, default=0, help="grow boxes by this many px when matching")
    ap.add_argument("--ap-floor", type=float, default=0.05,
                    help="ignore boxes below this conf for AP; YOLO at conf~0 carpets the frame (default 0.05)")
    ap.add_argument("--out", type=Path, help="output folder (default: <database folder>/eval)")
    args = ap.parse_args()
    out_dir = args.out or args.db.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_index, sizes = index_ground_truth(args.data_dir)
    print(f"ground truth: {len(gt_index)} frame files")
    gt_cache = {}

    # per (model, modality): det confidences + what they hit, image-level stats, target rows
    dets = defaultdict(lambda: {"conf": [], "kind": []})  # kind: 1 human, 2 animal, 0 nothing
    images = defaultdict(list)                              # (has_human, max_conf)
    targets = []                                            # rows for targets.csv
    missing_gt = 0

    con = open_db(args.db)
    rows = con.execute("SELECT i.episode, i.traj, i.frame, i.modality, d.model, d.boxes "
                       "FROM dets d JOIN images i ON i.id = d.image_id ORDER BY d.image_id, d.rowid")
    for episode, traj, frame, mod, model, blob in rows:
        key = (episode, traj, mod, frame)
        if key not in gt_index:
            missing_gt += 1
            continue
        if key not in gt_cache:
            gt_cache[key] = load_targets(gt_index[key], *sizes[key[:3]])
        names, xy, human, visible = gt_cache[key]
        conf, boxes, person = unpack_boxes(blob)
        conf, boxes = conf[person], boxes[person]  # other COCO classes are not scored
        hit = match(conf, boxes, xy, args.margin)
        kind = np.zeros(len(hit), int)  # frames can have no in-frame targets
        kind[hit >= 0] = np.where(human[hit[hit >= 0]], 1, 2)
        d = dets[(model, mod)]
        d["conf"].append(conf)
        d["kind"].append(kind)
        images[(model, mod)].append((human.any(), conf.max() if len(conf) else 0.0))
        best = {j: c for c, j in zip(conf, hit) if j >= 0}
        for j, name in enumerate(names):
            if human[j]:
                targets.append((model, mod, episode, traj, frame, name,
                                "visible" if visible[j] else "occluded", best.get(j, "")))
    con.close()
    print(f"scored {len(targets)} target rows")

    if missing_gt:
        print(f"warning: {missing_gt} (image, model) results had no ground truth file (skipped)")

    # targets per (model, modality): best conf (nan = missed) and visibility
    hits = defaultdict(lambda: {"conf": [], "vis": []})
    for model, mod, *_, vis, c in targets:
        hits[(model, mod)]["conf"].append(np.nan if c == "" else c)
        hits[(model, mod)]["vis"].append(vis == "visible")

    summary, curves = [], []
    for (model, mod) in sorted(dets):
        conf = np.concatenate(dets[(model, mod)]["conf"])
        kind = np.concatenate(dets[(model, mod)]["kind"])
        tconf = np.array(hits[(model, mod)]["conf"], float)
        tvis = np.array(hits[(model, mod)]["vis"], bool)
        img = np.array(images[(model, mod)], float)
        n_img, n_pos = len(img), len(tconf)
        floor = conf >= args.ap_floor
        ap_all = average_precision(conf[floor], kind[floor] == 1, n_pos)

        def at(t):
            keep = conf >= t
            tp, n_det = int((keep & (kind == 1)).sum()), int(keep.sum())
            found = np.nan_to_num(tconf, nan=-1) >= t
            rec = found.mean() if n_pos else np.nan
            prec = tp / n_det if n_det else np.nan
            pred, truth = img[:, 1] >= t, img[:, 0] > 0
            return {
                "recall": rec,
                "recall_visible": found[tvis].mean() if tvis.any() else np.nan,
                "recall_occluded": found[~tvis].mean() if (~tvis).any() else np.nan,
                "precision": prec,
                "f1": 2 * prec * rec / (prec + rec) if n_det and rec > 0 else 0.0,
                "fp_per_image": (n_det - tp) / n_img,
                "animal_hits": int((keep & (kind == 2)).sum()),
                "image_accuracy": float((pred == truth).mean()),
                "image_recall": float(pred[truth].mean()) if truth.any() else np.nan,
                "image_precision": float(truth[pred].mean()) if pred.any() else np.nan,
            }

        summary.append({"model": model, "modality": mod, "threshold": args.conf, "images": n_img,
                        "humans_in_frame": n_pos, "visible": int(tvis.sum()), "occluded": int((~tvis).sum()),
                        "AP": ap_all, "AP_floor": args.ap_floor, "margin": args.margin, **at(args.conf)})
        curves += [{"model": model, "modality": mod, "threshold": t, **at(t)} for t in THRESHOLDS]

    def write(name, rows):
        with open(out_dir / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)

    write("summary.csv", summary)
    write("curves.csv", curves)
    with open(out_dir / "targets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "modality", "episode", "traj", "frame", "target", "visibility", "hit_conf"])
        w.writerows(targets)

    cols = ["model", "modality", "AP", "recall", "recall_visible", "recall_occluded",
            "precision", "f1", "fp_per_image", "animal_hits", "image_accuracy"]
    print(f"\nthreshold {args.conf}, margin {args.margin}px, AP over conf >= {args.ap_floor}, humans = in-frame non-animal targets")
    print("  ".join(f"{c:>15}" for c in cols))
    for r in summary:
        print("  ".join(f"{r[c]:>15.3f}" if isinstance(r[c], float) else f"{str(r[c]):>15}" for c in cols))
    print(f"\nwrote {out_dir / 'summary.csv'}, curves.csv, targets.csv")


if __name__ == "__main__":
    main()
