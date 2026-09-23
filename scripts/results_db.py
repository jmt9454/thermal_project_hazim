"""
Pack classifier output JSONs (yolo_classifier_8_26s.py, qwen_*classifier.py) into one
SQLite file, so the viewer and evaluate.py can jump to any frame without loading
multi-GB JSONs into memory.  The JSONs stay the source of truth; rebuild the
database whenever they change.

Usage:
    python results_db.py ../results                        # every *.json -> ../results/detections.db
    python results_db.py ../results --conf-floor 0.01      # drop boxes below 0.01 (default: keep all)
    python results_db.py a.json b.json --out mine.db

Tables:
    images(id, episode, traj, frame, modality, file)
    dets(image_id, model, source, n, max_conf, boxes, detail, raw)
        boxes   float64 [n, 6] = conf, x1, y1, x2, y2, is_person    (fast, for matching)
        detail  zlib JSON list of the original detections, same order
                (label, box, conf, keypoints, parts, ...)             (for drawing)
        raw     the model's reply text, if the JSON had "<model>_raw"
    meta(key, value)
One dets row per (image, model), also when that model found nothing.
"""
import argparse
import json
import os
import sqlite3
import time
import zlib
from pathlib import Path, PurePosixPath

import numpy as np

SCHEMA = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE images(id INTEGER PRIMARY KEY, episode TEXT, traj TEXT, frame INTEGER,
                    modality TEXT, file TEXT, UNIQUE(episode, traj, frame, modality));
CREATE TABLE dets(image_id INTEGER, model TEXT, source TEXT, n INTEGER, max_conf REAL,
                  boxes BLOB, detail BLOB, raw TEXT, PRIMARY KEY(image_id, model));
"""
BATCH = 2000  # dets rows per insert, keeps memory flat while building


def episode_of(file, traj):
    """'.../EP_1/Traj_1/RGB/frame_5.png' with traj 'Traj_1' -> 'EP_1'; '' for the flat
    <data>/<episode>/<modality> layout, where the JSON's top-level key is the episode."""
    parts = PurePosixPath(file.replace("\\", "/")).parts
    return parts[-4] if len(parts) >= 4 and parts[-3] == traj else ""


def pack_boxes(dets):
    arr = np.array([[d["conf"], *d["box"], d.get("label", "person") == "person"] for d in dets],
                   dtype=np.float64)
    return arr.reshape(-1, 6).tobytes()


def unpack_boxes(blob):
    """-> conf [n], boxes [n, 4], is_person [n] (same order as the detail list)."""
    arr = np.frombuffer(blob, dtype=np.float64).reshape(-1, 6)
    return arr[:, 0], arr[:, 1:5], arr[:, 5] > 0


def unpack_detail(blob):
    return json.loads(zlib.decompress(blob))


def open_db(path):
    """Read-only connection, so a viewer can't lock or change the file."""
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def build(json_paths, out, conf_floor=0.0):
    tmp = out.with_name(out.name + ".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.executescript(SCHEMA)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    image_ids, seen, replaced = {}, set(), 0

    for path in json_paths:
        t0 = time.time()
        print(f"reading {path.name} ...", flush=True)
        data = json.loads(path.read_bytes())
        rows, n_dets = [], 0
        for traj, frames in data.items():
            for n, mods in frames.items():
                for mod, entry in mods.items():
                    episode = episode_of(entry.get("file", ""), traj)
                    frame = int(n) if n.isdigit() else n
                    key = (episode, traj, frame, mod)
                    if key not in image_ids:
                        cur = con.execute("INSERT INTO images(episode, traj, frame, modality, file) "
                                          "VALUES (?, ?, ?, ?, ?)", (*key, entry.get("file", "")))
                        image_ids[key] = cur.lastrowid
                    image_id = image_ids[key]
                    for model, dets in entry.items():
                        if not isinstance(dets, list):
                            continue  # "file", "<model>_raw"
                        dets = [d for d in dets if d["conf"] >= conf_floor]
                        if (image_id, model) in seen:
                            replaced += 1
                        seen.add((image_id, model))
                        rows.append((image_id, model, path.name, len(dets),
                                     max((d["conf"] for d in dets), default=None),
                                     pack_boxes(dets),
                                     zlib.compress(json.dumps(dets, separators=(",", ":")).encode(), 6),
                                     entry.get(f"{model}_raw")))
                        n_dets += len(dets)
                        if len(rows) >= BATCH:
                            con.executemany("INSERT OR REPLACE INTO dets VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
                            rows = []
        con.executemany("INSERT OR REPLACE INTO dets VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        con.commit()
        del data
        print(f"  {n_dets:,} boxes in {time.time() - t0:.0f}s", flush=True)

    meta = {"conf_floor": conf_floor, "sources": [p.name for p in json_paths],
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    con.executemany("INSERT INTO meta VALUES (?, ?)", [(k, json.dumps(v)) for k, v in meta.items()])
    con.commit()
    con.close()
    os.replace(tmp, out)  # only swap in a complete database
    if replaced:
        print(f"warning: {replaced} (image, model) pairs appeared in more than one JSON; the later file won")
    print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB, {len(image_ids):,} images)")


def main():
    ap = argparse.ArgumentParser(description="Classifier JSONs -> one SQLite detections database")
    ap.add_argument("paths", type=Path, nargs="+", help="result JSON files and/or folders of them")
    ap.add_argument("--out", type=Path, help="database file (default: <first folder>/detections.db)")
    ap.add_argument("--conf-floor", type=float, default=0.0,
                    help="drop detections below this confidence (default 0 = keep every box)")
    args = ap.parse_args()

    json_paths = []
    for p in args.paths:
        json_paths += sorted(p.glob("*.json")) if p.is_dir() else [p]
    if not json_paths:
        ap.error("no .json files found")
    first = args.paths[0]
    out = args.out or (first if first.is_dir() else first.parent) / "detections.db"
    build(json_paths, out, args.conf_floor)


if __name__ == "__main__":
    main()
