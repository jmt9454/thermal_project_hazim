"""
Detection viewer.

Any folder of frame images is a modality; its parent folder is a sequence, so both
    <data>/EP_1/Traj_1/RGB/frame_331866.png    (episode / trajectory / modality)
    <data>/ep1/rgb/frame_1_rgb.jpg             (older episode / modality layout)
work.  Detections come from a detections.db (results_db.py - any size, frames are
read on demand) or from small detection JSONs.  Every (source, model) pair gets
its own colour and can be toggled.

Ground truth: a frame_<n>.json next to an image (target points + visibility, see
evaluate.py) is drawn on the frame and used to mark each box as a hit or a false
positive with the same matching as evaluate.py (--margin px):
    thick solid box   hit on a person          dashed box   false positive
    filled dot        person found             hollow dot   person missed
    green = visible person, orange = occluded, purple square = animal
An annotations.json from annotator.py still shows its human / no-human label.

Tabs: Frames, plus Metrics / Curves / Timeline from evaluate.py's output folder
(<db folder>/eval is picked up automatically).  Click a curve to set the threshold,
click the timeline to jump to that frame.

Usage:
    python detection_viewer.py --data ../data --db ../results/detections.db
    python detection_viewer.py --data ../data/EP_1 --json small.json

Keys (Frames / Timeline tabs):
    <-/->  or  Up/Down     prev / next frame
    Home / End             first / last frame
    PageUp / PageDown      prev / next sequence
    1 / 2 / 3              show / hide modalities
    B  C  P  K  G  O       toggle boxes / conf labels / part boxes / keypoints /
                           ground truth / other (non-person) classes
    [  ]                   threshold down / up (1% of the slider range)
    Esc                    quit
"""

import argparse
import csv
import json
import math
import re
import sys
import tkinter as tk
from collections import OrderedDict, defaultdict
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk
except ImportError:
    sys.exit("Missing dependency: Pillow.  Install with:  pip install pillow")

from evaluate import load_targets, match
from results_db import open_db, unpack_boxes, unpack_detail

try:  # charts are optional; the Frames tab works without matplotlib
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError:
    Figure = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FRAME_RE = re.compile(r"frame[_-]?(\d+)", re.I)
MOD_ORDER = {"rgb": 0, "thermal": 1, "ir": 1, "rf": 2}
MAX_JSON_MB = 300          # bigger JSONs go through results_db.py
DEFAULT_THRESHOLD = 0.25
DEFAULT_MARGIN = 25

BG = "#1e1e1e"
PANEL_BG = "#252526"
CANVAS_BG = "#111"
COL_HUMAN = "#69f0ae"
COL_NO = "#ff5252"
COL_UNSET = "#888888"
COL_PARTIAL = "#ffd740"
GT_VISIBLE = "#00e676"
GT_OCCLUDED = "#ff9100"
GT_ANIMAL = "#e040fb"
GT_MISS = "#ff1744"
HEADER = 22  # px reserved above each grid tile for its title line
# one colour per source (JSON x model), reused cyclically
PALETTE = ["#ff5252", "#40c4ff", "#ffd740", "#69f0ae",
           "#e040fb", "#ff9100", "#b2ff59", "#f5f5f5"]

# COCO keypoint order used by YOLO pose models, and which pairs form the skeleton
KPT_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
             "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
             "left_wrist", "right_wrist", "left_hip", "right_hip",
             "left_knee", "right_knee", "left_ankle", "right_ankle"]
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
            (1, 3), (2, 4), (3, 5), (4, 6)]


# ----------------------------------------------------------------------------
# data helpers
# ----------------------------------------------------------------------------
def nat_key(name):
    m = re.search(r"(\d+)\s*$", name)
    return (int(m.group(1)) if m else 1 << 30, name)


def mod_key(m):
    return (MOD_ORDER.get(m.lower(), 9), m.lower())


def scan_sequences(data_root):
    """Every folder holding frame images is a modality; its parent is a sequence.
    -> {seq key: {"episode", "traj", "frames": [int], "images": {mod: {frame: Path}},
                  "gt": {mod: {frame: Path}}}}   (seq key = parent path relative to data_root)"""
    seqs = {}
    for p in data_root.rglob("*"):
        m = FRAME_RE.search(p.stem)
        if not m or p.suffix.lower() not in IMG_EXTS | {".json"}:
            continue
        mod_dir, seq_dir = p.parent, p.parent.parent
        if mod_dir == data_root:
            continue
        key = seq_dir.relative_to(data_root).as_posix() if seq_dir != data_root else seq_dir.name
        seq = seqs.setdefault(key, {"episode": seq_dir.parent.name, "traj": seq_dir.name,
                                    "frames": set(), "images": {}, "gt": {}})
        frame, mod = int(m.group(1)), mod_dir.name
        if p.suffix.lower() == ".json":
            seq["gt"].setdefault(mod, {})[frame] = p
        else:
            seq["images"].setdefault(mod, {})[frame] = p
            seq["frames"].add(frame)
    seqs = {k: v for k, v in seqs.items() if v["frames"]}
    for v in seqs.values():
        v["frames"] = sorted(v["frames"])
    return dict(sorted(seqs.items(), key=lambda kv: [nat_key(s) for s in kv[0].split("/")]))


def load_annotations(path):
    """Read annotator.py's annotations.json into {ep: {int frame: bool}}."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {ep: {int(k): bool(v) for k, v in fr.items()}
                for ep, fr in raw.items() if isinstance(fr, dict)}
    except (OSError, ValueError, AttributeError):
        return {}


def models_in(data):
    """All model keys (list-valued entries, e.g. 'yolov8', 'yolo26') in a detections JSON."""
    return sorted({k for e in data.values() for f in e.values() for m in f.values()
                   for k, v in m.items() if isinstance(v, list)})


def boxes_of(dets):
    """Detection dicts -> conf [n], boxes [n, 4], is_person [n], same order."""
    conf = np.array([d["conf"] for d in dets], float)
    boxes = np.array([d["box"] for d in dets], float).reshape(-1, 4)
    person = np.array([d.get("label", "person") == "person" for d in dets], bool)
    return conf, boxes, person


def fit(img, box_w, box_h):
    """Return (resized COPY of img, scale) so it fits inside box_w x box_h."""
    s = min(box_w / img.width, box_h / img.height) if box_w > 1 and box_h > 1 else 1.0
    new = (max(1, int(img.width * s)), max(1, int(img.height * s)))
    return img.resize(new, Image.BILINEAR), s


def best_grid(n, w, h, aspect):
    """Choose (cols, rows) for n tiles in a w x h area so each image is as big as
    possible.  aspect = image width / height."""
    best = (1, n, -1.0)
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        img_w = min(w / cols, (h / rows - HEADER) * aspect)  # width of a fitted image
        if img_w > best[2]:
            best = (cols, rows, img_w)
    return best[:2]


def get_font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def dashed_rect(d, box, color, width, dash=6):
    x1, y1, x2, y2 = box
    for a, b, horiz in (((x1, y1), (x2, y1), True), ((x1, y2), (x2, y2), True),
                        ((x1, y1), (x1, y2), False), ((x2, y1), (x2, y2), False)):
        start, end = (a[0], b[0]) if horiz else (a[1], b[1])
        t = start
        while t < end:
            u = min(t + dash, end)
            d.line((t, a[1], u, a[1]) if horiz else (a[0], t, a[0], u), fill=color, width=width)
            t += 2 * dash


# ----------------------------------------------------------------------------
# detection sources
# ----------------------------------------------------------------------------
class JsonSource:
    """One model's detections from one (small) JSON file, keyed by sequence folder name."""

    def __init__(self, json_path, json_no, data, model, color):
        self.tag = f"J{json_no}:{model}"
        self.file = json_path.name
        self.data, self.model, self.color = data, model, color
        self.var = tk.BooleanVar(value=True)
        self.max = max((d["conf"] for e in data.values() for f in e.values()
                        for m in f.values() for d in m.get(model, [])), default=0.0)

    def dets(self, seq, frame, mod):
        return self.data.get(seq["traj"], {}).get(str(frame), {}).get(mod, {}).get(self.model, [])

    def seq_boxes(self, seq):
        """-> {(frame, mod): (conf, boxes, is_person)} for every frame of the sequence."""
        out = {}
        for n, mods in self.data.get(seq["traj"], {}).items():
            for mod, entry in mods.items():
                if n.isdigit() and isinstance(entry.get(self.model), list):
                    out[(int(n), mod)] = boxes_of(entry[self.model])
        return out


class Database:
    """A detections.db from results_db.py, shared by one DbSource per model."""

    def __init__(self, path):
        self.path = Path(path)
        self.con = open_db(path)
        self.ids = {(e, t, f, m): i for i, e, t, f, m in
                    self.con.execute("SELECT id, episode, traj, frame, modality FROM images")}
        self.models = [r[0] for r in self.con.execute("SELECT DISTINCT model FROM dets ORDER BY model")]
        self.cache = OrderedDict()  # (image_id, model) -> detail list, the sliding window

    def image_id(self, seq, frame, mod):
        return (self.ids.get((seq["episode"], seq["traj"], frame, mod))
                or self.ids.get(("", seq["traj"], frame, mod)))

    def detail(self, image_id, model):
        key = (image_id, model)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        row = self.con.execute("SELECT detail FROM dets WHERE image_id = ? AND model = ?", key).fetchone()
        dets = unpack_detail(row[0]) if row else []
        self.cache[key] = dets
        if len(self.cache) > 400:
            self.cache.popitem(last=False)
        return dets


class DbSource:
    def __init__(self, db, model, color):
        self.db, self.model, self.color = db, model, color
        self.tag = model
        self.file = db.path.name
        self.var = tk.BooleanVar(value=True)
        self.max = db.con.execute("SELECT MAX(max_conf) FROM dets WHERE model = ?", (model,)).fetchone()[0] or 0.0

    def dets(self, seq, frame, mod):
        i = self.db.image_id(seq, frame, mod)
        return self.db.detail(i, self.model) if i else []

    def seq_boxes(self, seq):
        rows = self.db.con.execute(
            "SELECT i.frame, i.modality, d.boxes FROM images i JOIN dets d ON d.image_id = i.id "
            "WHERE i.traj = ? AND i.episode IN (?, '') AND d.model = ?",
            (seq["traj"], seq["episode"], self.model))
        return {(f, m): unpack_boxes(b) for f, m, b in rows}


# ----------------------------------------------------------------------------
# app
# ----------------------------------------------------------------------------
class Viewer(tk.Tk):
    def __init__(self, data_dir, json_paths, db_path=None, eval_dir=None):
        super().__init__()
        self.title("thermal_hazim detection viewer")
        self.geometry("1500x950")
        self.configure(bg=BG)

        self.seqs, self.seq_names, self.modalities = {}, [], []
        self.ann = {}          # human / no-human labels from annotations.json
        self.sources = []      # JsonSource / DbSource
        self.dbs = []
        self.frame_idx = 0     # position within the current sequence's frames
        self._photos, self._cache, self._pending, self._prefetch = [], {}, None, None
        self.gt = {}           # (mod, frame) -> (names, xy, human, visible) for the current sequence
        self.hits = {}         # (id(source), mod, frame) -> (conf, hit, is_person)
        self.sizes = {}        # (seq, mod) -> image size
        self.ev = None         # evaluate.py output: summary / curves / targets

        self.data_var = tk.StringVar()
        self.seq = tk.StringVar()
        self.thr = tk.DoubleVar(value=DEFAULT_THRESHOLD)
        self.margin = tk.IntVar(value=DEFAULT_MARGIN)
        self.full_range = tk.BooleanVar(value=False)
        self.mod_vars = {}     # modality -> BooleanVar (visible in grid?)
        self.show = {k: tk.BooleanVar(value=v) for k, v in
                     (("boxes", True), ("conf", True), ("parts", True), ("kpts", True),
                      ("gt", True), ("other", False))}
        self.font = get_font(12)

        self._build_ui()
        self._bind_keys()
        if data_dir:
            self.set_data_dir(data_dir)
        if db_path:
            self.open_db(db_path)
        for p in json_paths:
            self.add_json(p)
        eval_dir = eval_dir or (Path(db_path).parent / "eval" if db_path else None)
        if eval_dir and Path(eval_dir).is_dir():
            self.load_eval(eval_dir)

    # ---------------- UI ----------------
    def _build_ui(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        sep = lambda parent: ttk.Separator(parent, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8)

        # row 1: data folder, detection sources, eval folder, sequence
        top = ttk.Frame(self, padding=(6, 5))
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="Data:").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.data_var, width=40,
                  state="readonly").pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Browse…", command=self.browse_data).pack(side=tk.LEFT)
        sep(top)
        ttk.Button(top, text="Open DB…", command=self.browse_db).pack(side=tk.LEFT)
        ttk.Button(top, text="Add JSON…", command=self.browse_json).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Clear sources", command=self.clear_sources).pack(side=tk.LEFT)
        ttk.Button(top, text="Open eval…", command=self.browse_eval).pack(side=tk.LEFT, padx=4)
        sep(top)
        ttk.Label(top, text="Sequence:").pack(side=tk.LEFT)
        self.seq_menu = ttk.OptionMenu(top, self.seq, "")
        self.seq_menu.pack(side=tk.LEFT, padx=2)

        # row 2: which modalities are in the grid, what to draw, matching margin, ground truth
        row2 = ttk.Frame(self, padding=(6, 2))
        row2.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(row2, text="Show:").pack(side=tk.LEFT)
        self.mod_bar = ttk.Frame(row2)  # filled in set_data_dir()
        self.mod_bar.pack(side=tk.LEFT)
        sep(row2)
        ttk.Label(row2, text="Draw:").pack(side=tk.LEFT)
        for key, text in (("boxes", "boxes [B]"), ("conf", "conf [C]"), ("parts", "parts [P]"),
                          ("kpts", "keypoints [K]"), ("gt", "ground truth [G]"), ("other", "other classes [O]")):
            ttk.Checkbutton(row2, text=text, variable=self.show[key],
                            command=self.refresh).pack(side=tk.LEFT, padx=2)
        sep(row2)
        ttk.Label(row2, text="Match margin px:").pack(side=tk.LEFT)
        spin = ttk.Spinbox(row2, from_=0, to=200, increment=5, width=5, textvariable=self.margin,
                           command=self.rematch)
        spin.pack(side=tk.LEFT, padx=2)
        spin.bind("<Return>", lambda _e: self.rematch())    # typed values apply on Enter
        spin.bind("<FocusOut>", lambda _e: self.rematch())  # ... or when leaving the box
        self.gt_label = tk.Label(row2, text="", bg=BG, font=("Segoe UI", 11, "bold"))
        self.gt_label.pack(side=tk.RIGHT, padx=8)

        # row 3: global threshold slider
        row3 = ttk.Frame(self, padding=(6, 2))
        row3.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(row3, text="Threshold [ / ]:").pack(side=tk.LEFT)
        ttk.Checkbutton(row3, text="full 0–1 range", variable=self.full_range,
                        command=self.update_range).pack(side=tk.RIGHT)
        self.range_label = ttk.Label(row3, width=12)
        self.range_label.pack(side=tk.RIGHT)
        self.thr_label = ttk.Label(row3, width=8, font=("Consolas", 11, "bold"))
        self.thr_label.pack(side=tk.RIGHT, padx=6)
        self.scale = ttk.Scale(row3, from_=0.0, to=1.0, variable=self.thr, takefocus=0,
                               command=lambda _v: self.schedule_render(list_too=True))
        self.scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        self.status = ttk.Label(
            self, anchor="w", padding=(6, 2),
            text="←/→ frame   PgUp/PgDn sequence   1/2/3 modality   B/C/P/K/G/O draw toggles   "
                 "[ ] threshold   |   solid = hit, dashed = false positive, hollow dot = missed person")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True)
        self.nb.bind("<<NotebookTabChanged>>", lambda _e: self.update_charts())

        # Frames tab: canvas | side panel (sources + frame list)
        body = ttk.Frame(self.nb)
        self.nb.add(body, text="Frames")
        self.canvas = tk.Canvas(body, bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _e: self.schedule_render())

        side = ttk.Frame(body, width=300, padding=(6, 4))
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)
        ttk.Label(side, text="Sources").pack(anchor="w")
        self.src_frame = tk.Frame(side, bg=PANEL_BG)
        self.src_frame.pack(fill=tk.X, pady=(2, 8))
        self.list_title = ttk.Label(side, text="Frames")
        self.list_title.pack(anchor="w")
        lb_frame = ttk.Frame(side)
        lb_frame.pack(fill=tk.BOTH, expand=True)
        self.listbox = tk.Listbox(lb_frame, bg=PANEL_BG, fg="#ddd",
                                  font=("Consolas", 10), selectbackground="#0a84ff",
                                  activestyle="none", exportselection=False)
        sb = ttk.Scrollbar(lb_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)

        self._build_metrics_tab()
        self._build_curves_tab()
        self._build_timeline_tab()

    def _bind_keys(self):
        def key(f, nav=False):
            """Ignore keys typed into text fields; arrows don't navigate from the Metrics table."""
            def handler(e):
                if isinstance(e.widget, (tk.Entry, ttk.Entry)) or (nav and self.nb.index("current") == 1):
                    return None
                f()
                return "break"
            return handler
        self.bind("<Left>", key(lambda: self.step(-1), nav=True))
        self.bind("<Right>", key(lambda: self.step(1), nav=True))
        self.bind("<Up>", key(lambda: self.step(-1), nav=True))
        self.bind("<Down>", key(lambda: self.step(1), nav=True))
        self.bind("<Home>", key(lambda: self.goto(0), nav=True))
        self.bind("<End>", key(lambda: self.goto(len(self._frames()) - 1), nav=True))
        self.bind("<Prior>", key(lambda: self.step_seq(-1), nav=True))  # PageUp
        self.bind("<Next>", key(lambda: self.step_seq(1), nav=True))    # PageDown
        for i in range(1, 10):
            self.bind(str(i), key(lambda i=i: self.toggle_mod(i - 1)))
        for k, name in (("b", "boxes"), ("c", "conf"), ("p", "parts"), ("k", "kpts"),
                        ("g", "gt"), ("o", "other")):
            self.bind(k, key(lambda n=name: self.toggle_draw(n)))
            self.bind(k.upper(), key(lambda n=name: self.toggle_draw(n)))
        self.bind("[", key(lambda: self.nudge(-1)))
        self.bind("]", key(lambda: self.nudge(1)))
        self.bind("<Escape>", lambda e: self.destroy())

    # ---------------- loading ----------------
    def browse_data(self):
        p = filedialog.askdirectory(title="Data folder (episodes / trajectories / modalities)",
                                    initialdir=self.data_var.get() or ".")
        if p:
            self.set_data_dir(p)

    def browse_db(self):
        p = filedialog.askopenfilename(title="detections.db from results_db.py",
                                       filetypes=[("SQLite", "*.db"), ("All", "*.*")])
        if p:
            self.open_db(p)
            if (Path(p).parent / "eval").is_dir() and self.ev is None:
                self.load_eval(Path(p).parent / "eval")

    def browse_json(self):
        start = Path(self.data_var.get()).parent if self.data_var.get() else "."
        for p in filedialog.askopenfilenames(title="Detection JSON file(s)", initialdir=start,
                                             filetypes=[("JSON", "*.json"), ("All", "*.*")]):
            self.add_json(p)

    def browse_eval(self):
        p = filedialog.askdirectory(title="evaluate.py output folder (summary.csv, curves.csv, targets.csv)")
        if p:
            self.load_eval(p)

    def set_data_dir(self, path):
        path = Path(path).resolve()
        self.status.configure(text=f"scanning {path} ...")
        self.update_idletasks()
        seqs = scan_sequences(path) if path.is_dir() else {}
        if not seqs:
            self.status.configure(text=f"no frame images under {path}")
            return
        self.data_var.set(str(path))
        self.seqs, self.seq_names = seqs, list(seqs)
        self.modalities = sorted({m for s in seqs.values() for m in s["images"]}, key=mod_key)
        self._cache.clear()
        self.sizes.clear()

        for w in self.mod_bar.winfo_children():
            w.destroy()
        self.mod_vars = {m: tk.BooleanVar(value=True) for m in self.modalities}
        for i, m in enumerate(self.modalities, start=1):
            ttk.Checkbutton(self.mod_bar, text=f"{m.upper()} [{i}]", variable=self.mod_vars[m],
                            command=self.refresh).pack(side=tk.LEFT, padx=2)

        menu = self.seq_menu["menu"]
        menu.delete(0, tk.END)
        for name in self.seq_names:
            menu.add_command(label=name, command=lambda n=name: (self.seq.set(n), self.load_seq()))
        self.seq.set(self.seq_names[0])

        self.ann = {}  # annotator.py saves next to the data folder by default
        for p in (path.parent / "annotations.json", path / "annotations.json"):
            if p.is_file():
                self.ann = load_annotations(p)
                break
        n_gt = sum(len(f) for s in seqs.values() for f in s["gt"].values())
        self.status.configure(text=f"{len(seqs)} sequences, {n_gt} ground truth files")
        self.load_seq()

    def open_db(self, path):
        try:
            db = Database(path)
        except Exception as e:
            self.status.configure(text=f"could not open {path}: {e}")
            return
        self.dbs.append(db)
        for model in db.models:
            self.sources.append(DbSource(db, model, PALETTE[len(self.sources) % len(PALETTE)]))
        self._sources_changed()

    def add_json(self, path):
        path = Path(path)
        if path.stat().st_size > MAX_JSON_MB * 1e6:
            self.status.configure(text=f"{path.name} is {path.stat().st_size / 1e9:.1f} GB - pack it with "
                                       f"results_db.py and use Open DB… instead")
            return
        try:
            data = json.loads(path.read_bytes())
        except (OSError, ValueError) as e:
            self.status.configure(text=f"could not read {path.name}: {e}")
            return
        json_no = len({id(s.data) for s in self.sources if isinstance(s, JsonSource)}) + 1
        for model in models_in(data):  # one source per model inside the file
            color = PALETTE[len(self.sources) % len(PALETTE)]
            self.sources.append(JsonSource(path, json_no, data, model, color))
        self._sources_changed()

    def clear_sources(self):
        self.sources, self.dbs = [], []
        self._sources_changed()

    def _sources_changed(self):
        """Redraw the coloured source checkboxes, re-match, update slider range and list."""
        for w in self.src_frame.winfo_children():
            w.destroy()
        for s in self.sources:
            tk.Checkbutton(self.src_frame, text=f"{s.tag}   max {s.max:.3f}\n{s.file}",
                           variable=s.var, command=self.refresh, anchor="w",
                           justify="left", wraplength=270, fg=s.color, bg=PANEL_BG,
                           selectcolor=PANEL_BG, activebackground="#333",
                           activeforeground=s.color, font=("Consolas", 9)).pack(fill=tk.X)
        self.rematch()
        self.update_range()

    def update_range(self):
        """Slider spans 0..(highest conf in the loaded sources), or 0..1 if ticked."""
        data_max = max((s.max for s in self.sources), default=1.0)
        top = 1.0 if self.full_range.get() else max(data_max, 1e-3)
        self.scale.configure(to=top)
        if self.thr.get() > top:
            self.thr.set(top)
        self.range_label.configure(text=f"range 0–{top:.3f}")
        self.refresh()

    # ---------------- ground truth + matching for the current sequence ----------------
    def load_seq(self):
        self.frame_idx = 0
        self.rematch()
        self.update_charts()

    def _size(self, mod):
        key = (self.seq.get(), mod)
        if key not in self.sizes:
            img = next(iter(self._seq()["images"].get(mod, {}).values()), None)
            self.sizes[key] = Image.open(img).size if img else (1920, 1080)
        return self.sizes[key]

    def rematch(self):
        """Load the sequence's ground truth and match every source's boxes to it."""
        self.gt, self.hits = {}, {}
        seq = self._seq()
        if seq is None:
            return
        margin = self._margin()
        self.status.configure(text=f"matching {self.seq.get()} ...")
        self.update_idletasks()
        for mod, frames in seq["gt"].items():
            w, h = self._size(mod)
            for frame, path in frames.items():
                try:
                    self.gt[(mod, frame)] = load_targets(path, w, h)
                except (OSError, ValueError, KeyError):
                    pass  # not a ground truth file
        for s in self.sources:
            for (frame, mod), (conf, boxes, person) in s.seq_boxes(seq).items():
                hit = np.full(len(conf), -1)
                g = self.gt.get((mod, frame))
                if g is not None:
                    idx = np.flatnonzero(person)
                    hit[idx] = match(conf[idx], boxes[idx], g[1], margin)
                self.hits[(id(s), mod, frame)] = (conf, hit, person)
        n_gt = len({f for _m, f in self.gt})
        self.status.configure(text=f"{self.seq.get()}: {len(self._frames())} frames, "
                                   f"{n_gt} with ground truth, margin {margin:g}px")
        self.refresh()

    def _frame_stats(self, frame, mods, srcs, t):
        """Union over visible modalities and active sources at threshold t.
        -> (visible humans, found visible, occluded, found occluded, false positives, max conf)"""
        vis, occ, found, fp, best = set(), set(), set(), 0, None
        for m in mods:
            g = self.gt.get((m, frame))
            if g is not None:
                names, _xy, human, visible = g
                vis |= {n for n, hu, v in zip(names, human, visible) if hu and v}
                occ |= {n for n, hu, v in zip(names, human, visible) if hu and not v}
            for s in srcs:
                h = self.hits.get((id(s), m, frame))
                if h is None:
                    continue
                conf, hit, person = h
                if len(conf):
                    best = max(best or 0.0, float(conf.max()))
                keep = (conf >= t) & person
                if g is None:
                    continue
                for j in hit[keep & (hit >= 0)]:
                    if g[2][j]:
                        found.add(g[0][j])
                fp += int((keep & ((hit < 0) | ~g[2][np.maximum(hit, 0)])).sum()) if len(g[2]) else int(keep.sum())
        return len(vis), len(found & vis), len(occ), len(found & occ), fp, best

    # ---------------- helpers ----------------
    def _margin(self):
        """Current matching margin; a half-typed spinbox value keeps the last good one."""
        try:
            self._last_margin = max(0.0, float(self.margin.get()))
        except (tk.TclError, ValueError):
            pass
        return getattr(self, "_last_margin", float(DEFAULT_MARGIN))

    def _seq(self):
        return self.seqs.get(self.seq.get())

    def _frames(self):
        seq = self._seq()
        return seq["frames"] if seq else []

    def _cur_frame(self):
        fr = self._frames()
        return fr[self.frame_idx] if fr else None

    def _image_path(self, modality, frame):
        return self._seq()["images"].get(modality, {}).get(frame)

    def _visible_mods(self):
        return [m for m in self.modalities if self.mod_vars[m].get()]

    def _active_sources(self):
        return [s for s in self.sources if s.var.get()]

    def _ann(self, frame):
        seq = self._seq()
        return self.ann.get(seq["traj"], {}).get(frame) if seq else None  # True / False / None

    def _load(self, path):
        if path not in self._cache:
            if len(self._cache) > 24:
                self._cache.clear()
            self._cache[path] = Image.open(path).convert("RGB")
        return self._cache[path]

    # ---------------- navigation / toggles ----------------
    def step(self, d):
        fr = self._frames()
        if fr:
            self.goto((self.frame_idx + d) % len(fr))

    def goto(self, i):
        fr = self._frames()
        if fr:
            self.frame_idx = max(0, min(i, len(fr) - 1))
            self._select_list_row()
            self.render()
            self.update_charts(timeline_only=True)

    def goto_frame(self, frame):
        """Jump to the frame number closest to `frame` in the current sequence."""
        fr = self._frames()
        if fr:
            self.goto(int(np.argmin(np.abs(np.array(fr) - frame))))

    def step_seq(self, d):
        if self.seq_names:
            i = (self.seq_names.index(self.seq.get()) + d) % len(self.seq_names)
            self.seq.set(self.seq_names[i])
            self.load_seq()

    def _on_list_select(self, _e):
        sel = self.listbox.curselection()
        if sel and sel[0] != self.frame_idx:
            self.goto(sel[0])

    def toggle_mod(self, i):
        if i < len(self.modalities):
            v = self.mod_vars[self.modalities[i]]
            v.set(not v.get())
            self.refresh()

    def toggle_draw(self, name):
        self.show[name].set(not self.show[name].get())
        self.render()

    def nudge(self, d):
        top = float(self.scale.cget("to"))
        self.set_threshold(self.thr.get() + d * top / 100)

    def set_threshold(self, t):
        if t > float(self.scale.cget("to")):
            self.full_range.set(True)
            self.update_range()
        self.thr.set(min(float(self.scale.cget("to")), max(0.0, t)))
        self.refresh()

    def refresh(self):
        """Something changed that affects the frame list as well as the image."""
        self._fill_list()
        self.render()
        self.update_charts()

    # ---------------- frame list ----------------
    def _fill_list(self):
        """One row per frame: visible people found / in frame, false positives, max conf."""
        self.listbox.delete(0, tk.END)
        mods, srcs, t = self._visible_mods(), self._active_sources(), self.thr.get()
        has_gt = bool(self.gt)
        self.list_title.configure(text="Frames   found/visible  FP  frame  max conf" if has_gt
                                  else "Frames   (label, max conf)")
        for i, f in enumerate(self._frames()):
            nv, fv, _no, _fo, fp, best = self._frame_stats(f, mods, srcs, t)
            conf = f" {best:.3f}" if best is not None else ""
            if has_gt:
                text = f" {fv:>2}/{nv:<2} {fp:>3}  {f:<7}{conf}"
                color = COL_UNSET if nv == 0 else COL_HUMAN if fv == nv else COL_PARTIAL if fv else COL_NO
            else:
                tag, color = {True: ("H", COL_HUMAN), False: ("-", COL_NO)}.get(self._ann(f), ("?", COL_UNSET))
                text = f" {tag}  frame {f:<7}{conf}"
            self.listbox.insert(tk.END, text)
            self.listbox.itemconfig(i, foreground=color)
        self._select_list_row()

    def _select_list_row(self):
        self.listbox.selection_clear(0, tk.END)
        if self._frames():
            self.listbox.selection_set(self.frame_idx)
            self.listbox.see(self.frame_idx)

    # ---------------- rendering ----------------
    def schedule_render(self, list_too=False):
        """Debounce: slider drags / window resizes redraw at most every 30 ms."""
        if self._pending:
            self.after_cancel(self._pending)
        self._pending = self.after(30, self.refresh if list_too else self.render)

    def render(self):
        self._pending = None
        self.thr_label.configure(text=f"{self.thr.get():.4f}")
        c = self.canvas
        c.delete("all")
        self._photos = []  # keep PhotoImages alive while shown
        cw, ch = max(c.winfo_width(), 2), max(c.winfo_height(), 2)

        frame = self._cur_frame()
        if frame is None:
            c.create_text(20, 20, anchor="nw", fill="#aaa",
                          text="Pick a data folder (Browse…) and open a detections.db or add JSONs.")
            return
        self._refresh_header(frame)
        mods = self._visible_mods()
        if not mods:
            c.create_text(cw // 2, ch // 2, fill="#aaa", text="No modality shown — press 1/2/3")
            return

        seq, t = self._seq(), self.thr.get()
        paths = [self._image_path(m, frame) for m in mods]
        first = next((p for p in paths if p), None)
        aspect = (lambda im: im.width / im.height)(self._load(first)) if first else 4 / 3
        cols, rows = best_grid(len(mods), cw, ch, aspect)
        tw, th = cw / cols, ch / rows  # tile size

        for i, (m, path) in enumerate(zip(mods, paths)):
            x0, y0 = (i % cols) * tw, (i // cols) * th
            title = [(m.upper(), "#ddd")]
            if path is None:
                c.create_text(x0 + tw / 2, y0 + th / 2, fill="#aaa",
                              text=f"{m} missing for frame {frame}")
            else:
                img, s = fit(self._load(path), tw - 4, th - HEADER - 4)
                g = self.gt.get((m, frame))
                found = set()
                for src in self._active_sources():
                    conf, hit, _person = self.hits.get((id(src), m, frame), (np.zeros(0), np.zeros(0, int), None))
                    n, n_hit, got = self._draw_dets(img, s, src.dets(seq, frame, m), hit, g, src.color, t)
                    found |= got
                    title.append((f"{src.tag}: {n}" + (f" ({n_hit} hit)" if g is not None else ""), src.color))
                if g is not None:
                    nv = int((g[2] & g[3]).sum())
                    title.insert(1, (f"GT {len(found & set(np.flatnonzero(g[2] & g[3])))}/{nv} visible", GT_VISIBLE))
                    if self.show["gt"].get():
                        self._draw_gt(img, s, g, found)
                self._photos.append(ImageTk.PhotoImage(img))
                c.create_image(x0 + tw / 2, y0 + HEADER + (th - HEADER) / 2,
                               image=self._photos[-1], anchor="center")
            # tile title: modality, ground truth, then "source: #shown" in each source's colour
            x = x0 + 6
            for text, color in title:
                if x > x0 + tw - 40:  # out of room in this tile
                    break
                item = c.create_text(x, y0 + 4, anchor="nw", text=text, fill=color,
                                     font=("Segoe UI", 10, "bold"))
                x = c.bbox(item)[2] + 12
        self._schedule_prefetch()

    def _schedule_prefetch(self):
        """Sliding window: after drawing, pull the neighbouring frames from the database."""
        if self._prefetch:
            self.after_cancel(self._prefetch)
        self._prefetch = self.after(150, self._prefetch_neighbours)

    def _prefetch_neighbours(self):
        self._prefetch = None
        fr, seq = self._frames(), self._seq()
        for d in (1, -1, 2):
            if fr:
                f = fr[(self.frame_idx + d) % len(fr)]
                for s in self._active_sources():
                    if isinstance(s, DbSource):
                        for m in self._visible_mods():
                            s.dets(seq, f, m)

    def _draw_dets(self, img, s, dets, hit, g, color, t):
        """Draw every detection with conf >= t onto img (image scale s).
        Solid thick = hit on a person, dashed = false positive (or animal).
        Returns (#drawn, #person hits, set of target indices hit)."""
        d = ImageDraw.Draw(img)
        sc = lambda xs: [v * s for v in xs]
        shown, n_hit, got = 0, 0, set()
        order = sorted(range(len(dets)), key=lambda i: dets[i]["conf"])  # weakest first, strongest on top
        for i in order:
            det = dets[i]
            if det["conf"] < t:
                continue
            person = det.get("label", "person") == "person"
            if not person and not self.show["other"].get():
                continue
            shown += 1
            j = int(hit[i]) if i < len(hit) else -1
            is_hit = g is not None and j >= 0 and g[2][j]
            if is_hit:
                n_hit += 1
                got.add(j)
            if person and self.show["kpts"].get() and "keypoints" in det:
                pts = [det["keypoints"].get(n) for n in KPT_NAMES]
                # (0, 0) means the model did not place that keypoint
                ok = [p is not None and p[2] >= t and (p[0] > 0 or p[1] > 0) for p in pts]
                for a, b in SKELETON:
                    if ok[a] and ok[b]:
                        d.line(sc(pts[a][:2]) + sc(pts[b][:2]), fill=color, width=1)
                for p, good in zip(pts, ok):
                    if good:
                        x, y = sc(p[:2])
                        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color)
            if person and self.show["parts"].get():
                for part in det.get("parts", {}).values():
                    if part["conf"] >= t:
                        d.rectangle(sc(part["box"]), outline=color, width=1)
            if self.show["boxes"].get():
                box = sc(det["box"])
                if not person:
                    d.rectangle(box, outline=color, width=1)
                elif is_hit or g is None:
                    d.rectangle(box, outline=color, width=3 if is_hit else 2)
                else:
                    dashed_rect(d, box, color, 2)
            if self.show["conf"].get():
                x, y = sc(det["box"][:2])
                label = f"{det['conf']:.3f}"
                if not person:
                    label += f" {det.get('label', '')}"
                elif g is not None and j >= 0 and not g[2][j]:
                    label += " animal"
                h = d.textbbox((0, 0), label, font=self.font)[3] + 2
                y = max(0, y - h)  # just above the box
                d.rectangle(d.textbbox((x + 1, y), label, font=self.font), fill="black")
                d.text((x + 1, y), label, fill=color, font=self.font)
        return shown, n_hit, got

    def _draw_gt(self, img, s, g, found):
        """Target points: filled = found by an active source, hollow = missed.
        The faint square is the matching margin around the point."""
        d = ImageDraw.Draw(img)
        names, xy, human, visible = g
        r, mg = 6, self._margin() * s
        for j, (x, y) in enumerate(xy * s):
            if not human[j]:
                d.rectangle([x - r, y - r, x + r, y + r], outline=GT_ANIMAL, width=2)
                continue
            col = GT_VISIBLE if visible[j] else GT_OCCLUDED
            if mg > r:
                d.rectangle([x - mg, y - mg, x + mg, y + mg], outline=col, width=1)
            if j in found:
                d.ellipse([x - r, y - r, x + r, y + r], fill=col, outline="black")
            else:
                d.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2],
                          outline=GT_MISS if visible[j] else col, width=3)

    def _refresh_header(self, frame):
        mods, srcs = self._visible_mods(), self._active_sources()
        if self.gt:
            nv, fv, no, fo, fp, _ = self._frame_stats(frame, mods, srcs, self.thr.get())
            text = f"found {fv}/{nv} visible · {fo}/{no} occluded · {fp} FP"
            color = COL_UNSET if nv == 0 else COL_HUMAN if fv == nv else COL_PARTIAL if fv else COL_NO
        else:
            ann = self._ann(frame)
            text = {True: "GT: HUMAN", False: "GT: NO HUMAN"}.get(ann, "GT: —" if self.ann else "")
            color = {True: COL_HUMAN, False: COL_NO}.get(ann, COL_UNSET)
        self.gt_label.configure(text=text, fg=color)
        pos = f"{self.frame_idx + 1}/{len(self._frames())}"
        self.title(f"thermal_hazim detection viewer  [{self.seq.get()}  frame {frame}  {pos}]")

    # ---------------- eval tabs ----------------
    def load_eval(self, folder):
        folder = Path(folder)
        need = {n: folder / f"{n}.csv" for n in ("summary", "curves", "targets")}
        missing = [n for n, p in need.items() if not p.is_file()]
        if missing:
            self.status.configure(text=f"{folder}: missing {', '.join(m + '.csv' for m in missing)}")
            return
        self.status.configure(text=f"loading eval from {folder} ...")
        self.update_idletasks()
        with open(need["summary"], newline="") as f:
            summary = list(csv.DictReader(f))
        with open(need["curves"], newline="") as f:
            curves = defaultdict(list)
            for r in csv.DictReader(f):
                curves[(r["model"], r["modality"])].append(r)
        targets = defaultdict(lambda: defaultdict(list))  # (model, mod, episode, traj) -> frame -> [(vis, conf)]
        with open(need["targets"], newline="") as f:
            for r in csv.DictReader(f):
                c = float(r["hit_conf"]) if r["hit_conf"] else -1.0
                targets[(r["model"], r["modality"], r["episode"], r["traj"])][int(r["frame"])].append(
                    (r["visibility"] == "visible", c))
        self.ev = {"folder": folder, "summary": summary, "curves": curves, "targets": targets}

        margin = summary[0].get("margin") if summary else None
        if margin not in (None, ""):  # match the frame view to how the tables were scored
            self.margin.set(int(float(margin)))
            self.rematch()
        self._fill_metrics()
        self._fill_series()
        models = sorted({k[0] for k in targets})
        mods = sorted({k[1] for k in targets}, key=mod_key)
        self.tl_model.configure(values=models)
        self.tl_mod.configure(values=mods)
        if models and self.tl_model.get() not in models:
            self.tl_model.set(models[0])
        if mods and self.tl_mod.get() not in mods:
            self.tl_mod.set(mods[0])
        self.status.configure(text=f"eval: {folder}  ({len(summary)} model × modality rows"
                                   + (f", margin {margin}px" if margin else "") + ")")
        self.update_charts()

    def _build_metrics_tab(self):
        page = ttk.Frame(self.nb, padding=6)
        self.nb.add(page, text="Metrics")
        self.metrics_info = ttk.Label(page, text="Open an eval folder (evaluate.py output) to see metrics.")
        self.metrics_info.pack(anchor="w")
        frame = ttk.Frame(page)
        frame.pack(fill=tk.BOTH, expand=True, pady=4)
        self.tree = ttk.Treeview(frame, show="headings")
        ysb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.tree.yview)
        xsb = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.tree.bind("<Double-1>", self._on_metric_double)
        ttk.Label(page, text="Click a column header to sort · double-click a row to plot only that "
                             "model × modality on the Curves tab").pack(anchor="w")

    def _fill_metrics(self):
        rows = self.ev["summary"]
        cols = list(rows[0]) if rows else []
        self.tree.configure(columns=cols)
        for c in cols:
            self.tree.heading(c, text=c, command=lambda c=c: self._sort_metrics(c))
            self.tree.column(c, width=110 if c in ("model", "modality") else 90, anchor="e", stretch=False)
        self.tree.delete(*self.tree.get_children())
        for r in rows:
            self.tree.insert("", tk.END, values=[r[c] for c in cols])
        r0 = rows[0] if rows else {}
        self.metrics_info.configure(text=f"{self.ev['folder'] / 'summary.csv'}   threshold {r0.get('threshold', '?')}"
                                         f", margin {r0.get('margin', '?')}px, AP over conf ≥ {r0.get('AP_floor', '?')}")
        self._sort_dir = {}

    def _sort_metrics(self, col):
        rows = [(self.tree.set(k, col), k) for k in self.tree.get_children("")]
        num = lambda v: (0, float(v)) if v not in ("", "nan") else (1, 0.0)
        try:
            rows.sort(key=lambda r: num(r[0]), reverse=self._sort_dir.get(col, True))
        except ValueError:
            rows.sort(key=lambda r: r[0], reverse=self._sort_dir.get(col, False))
        self._sort_dir[col] = not self._sort_dir.get(col, True)
        for i, (_v, k) in enumerate(rows):
            self.tree.move(k, "", i)

    def _on_metric_double(self, _e):
        sel = self.tree.selection()
        if not sel or not self.ev:
            return
        model, mod = self.tree.set(sel[0], "model"), self.tree.set(sel[0], "modality")
        for key, var in self.series_vars.items():
            var.set(key == (model, mod))
        self.nb.select(2)

    def _build_curves_tab(self):
        page = ttk.Frame(self.nb, padding=4)
        self.nb.add(page, text="Curves")
        self.series_vars = {}
        left = ttk.Frame(page, width=220)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)
        ttk.Label(left, text="Series (model · modality)").pack(anchor="w")
        btns = ttk.Frame(left)
        btns.pack(anchor="w", pady=2)
        ttk.Button(btns, text="All", width=6, command=lambda: self._set_series(True)).pack(side=tk.LEFT)
        ttk.Button(btns, text="None", width=6, command=lambda: self._set_series(False)).pack(side=tk.LEFT, padx=4)
        self.series_box = ttk.Frame(left)
        self.series_box.pack(fill=tk.BOTH, expand=True)
        ttk.Label(left, text="Click a threshold plot to\nset the threshold.", foreground="#888").pack(anchor="w")
        if Figure is None:
            ttk.Label(page, text="matplotlib is not installed - charts unavailable").pack()
            self.curve_fig = None
            return
        self.curve_fig = Figure(figsize=(9, 6), dpi=100, facecolor=PANEL_BG)
        self.curve_axes = self.curve_fig.subplots(2, 2)
        self.curve_canvas = FigureCanvasTkAgg(self.curve_fig, master=page)
        self.curve_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.curve_canvas.mpl_connect("button_press_event", self._on_curve_click)

    def _fill_series(self):
        for w in self.series_box.winfo_children():
            w.destroy()
        self.series_vars = {}
        for key in sorted(self.ev["curves"], key=lambda k: (k[0], mod_key(k[1]))):
            var = tk.BooleanVar(value=True)
            self.series_vars[key] = var
            ttk.Checkbutton(self.series_box, text=f"{key[0]} · {key[1]}", variable=var,
                            command=self.update_charts).pack(anchor="w")

    def _set_series(self, on):
        for var in self.series_vars.values():
            var.set(on)
        self.update_charts()

    def _style_axes(self, ax, title, xlabel, ylabel):
        ax.set_facecolor(CANVAS_BG)
        ax.set_title(title, color="#ddd", fontsize=10)
        ax.set_xlabel(xlabel, color="#aaa", fontsize=9)
        ax.set_ylabel(ylabel, color="#aaa", fontsize=9)
        ax.tick_params(colors="#aaa", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#555")
        ax.grid(True, color="#333", linewidth=0.5)

    def _series_style(self, key):
        models = sorted({k[0] for k in self.ev["curves"]})
        mods = sorted({k[1] for k in self.ev["curves"]}, key=mod_key)
        colors = ["#40c4ff", "#ff5252", "#ffd740", "#69f0ae", "#e040fb", "#ff9100", "#b2ff59", "#f5f5f5"]
        return colors[models.index(key[0]) % len(colors)], ["-", "--", ":", "-."][mods.index(key[1]) % 4]

    def _draw_curves(self):
        f = lambda r, k: float(r[k]) if r[k] not in ("", "nan") else np.nan
        (a_rec, a_prec), (a_fp, a_pr) = self.curve_axes
        for ax in (a_rec, a_prec, a_fp, a_pr):
            ax.clear()
        # below the AP floor YOLO carpets the frame (recall -> 1 by chance), which flattens everything else
        floor = next((float(r["AP_floor"]) for r in self.ev["summary"] if r.get("AP_floor")), 0.0)
        self._style_axes(a_rec, "Recall (visible people)", "threshold", "recall")
        self._style_axes(a_prec, "Precision", "threshold", "precision")
        self._style_axes(a_fp, "False positives per image", "threshold", "FP / image")
        self._style_axes(a_pr, "Precision vs recall (all in-frame people)", "recall", "precision")
        if floor:
            a_rec.set_title(f"Recall (visible people) · thresholds ≥ {floor:g}", color="#ddd", fontsize=10)
        for key, var in self.series_vars.items():
            if not var.get():
                continue
            rows = [r for r in self.ev["curves"][key] if float(r["threshold"]) >= floor]
            t = [f(r, "threshold") for r in rows]
            col, ls = self._series_style(key)
            label = f"{key[0]} · {key[1]}"
            a_rec.plot(t, [f(r, "recall_visible") for r in rows], color=col, ls=ls, lw=1.5, label=label)
            a_prec.plot(t, [f(r, "precision") for r in rows], color=col, ls=ls, lw=1.5)
            a_fp.plot(t, [max(f(r, "fp_per_image"), 1e-4) for r in rows], color=col, ls=ls, lw=1.5)
            a_pr.plot([f(r, "recall") for r in rows], [f(r, "precision") for r in rows],
                      color=col, ls=ls, lw=1.5, marker=".", ms=4)
        a_fp.set_yscale("log")
        thr = self.thr.get()
        for ax in (a_rec, a_prec, a_fp):
            ax.axvline(thr, color="#fff", lw=1, alpha=0.6)
            ax.set_xlim(0, 1)
        a_rec.set_ylim(bottom=0)
        a_prec.set_ylim(0, 1.02)
        if a_rec.lines[:-1]:
            a_rec.legend(fontsize=7, facecolor=PANEL_BG, edgecolor="#555", labelcolor="#ddd")
        self.curve_fig.tight_layout()
        self.curve_canvas.draw_idle()

    def _on_curve_click(self, event):
        if event.inaxes in (self.curve_axes[0][0], self.curve_axes[0][1], self.curve_axes[1][0]) \
                and event.xdata is not None:
            self.set_threshold(round(float(event.xdata), 4))

    def _build_timeline_tab(self):
        page = ttk.Frame(self.nb, padding=4)
        self.nb.add(page, text="Timeline")
        bar = ttk.Frame(page)
        bar.pack(fill=tk.X)
        ttk.Label(bar, text="Model:").pack(side=tk.LEFT)
        self.tl_model = ttk.Combobox(bar, state="readonly", width=18)
        self.tl_model.pack(side=tk.LEFT, padx=4)
        ttk.Label(bar, text="Modality:").pack(side=tk.LEFT)
        self.tl_mod = ttk.Combobox(bar, state="readonly", width=8)
        self.tl_mod.pack(side=tk.LEFT, padx=4)
        ttk.Label(bar, text="People:").pack(side=tk.LEFT)
        self.tl_vis = ttk.Combobox(bar, state="readonly", width=9, values=["visible", "occluded", "all"])
        self.tl_vis.set("visible")
        self.tl_vis.pack(side=tk.LEFT, padx=4)
        for cb in (self.tl_model, self.tl_mod, self.tl_vis):
            cb.bind("<<ComboboxSelected>>", lambda _e: self.update_charts())
        ttk.Label(bar, text="  current sequence, at the current threshold · click to jump to a frame",
                  foreground="#888").pack(side=tk.LEFT)
        if Figure is None:
            ttk.Label(page, text="matplotlib is not installed - charts unavailable").pack()
            self.tl_fig = None
            return
        self.tl_fig = Figure(figsize=(9, 5), dpi=100, facecolor=PANEL_BG)
        self.tl_ax = self.tl_fig.add_subplot(111)
        self.tl_canvas = FigureCanvasTkAgg(self.tl_fig, master=page)
        self.tl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.tl_canvas.mpl_connect("button_press_event", self._on_timeline_click)

    def _draw_timeline(self):
        ax = self.tl_ax
        ax.clear()
        seq = self._seq()
        key = (self.tl_model.get(), self.tl_mod.get(), seq["episode"] if seq else "", seq["traj"] if seq else "")
        rows = self.ev["targets"].get(key) or self.ev["targets"].get((*key[:2], "", key[3]), {})
        self._style_axes(ax, f"{self.seq.get()}  ·  {key[0]} · {key[1]}  ·  {self.tl_vis.get()} people",
                         "frame", "people")
        if rows:
            want = self.tl_vis.get()
            t = self.thr.get()
            frames = sorted(rows)
            total = [sum(1 for v, _c in rows[f] if want == "all" or v == (want == "visible")) for f in frames]
            found = [sum(1 for v, c in rows[f] if (want == "all" or v == (want == "visible")) and c >= t)
                     for f in frames]
            ax.fill_between(frames, total, step="mid", color="#555", alpha=0.6, label="in frame")
            ax.fill_between(frames, found, step="mid", color="#69f0ae", alpha=0.9, label=f"found (conf ≥ {t:.2f})")
            n_t, n_f = sum(total), sum(found)
            ax.text(0.01, 0.97, f"found {n_f}/{n_t}" + (f" = {n_f / n_t:.1%}" if n_t else ""),
                    transform=ax.transAxes, va="top", color="#ddd", fontsize=9)
            ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor="#555", labelcolor="#ddd", loc="upper right")
            ax.set_ylim(bottom=0)
        else:
            ax.text(0.5, 0.5, "no eval rows for this sequence / model / modality",
                    transform=ax.transAxes, ha="center", color="#aaa")
        cur = self._cur_frame()
        if cur is not None:
            ax.axvline(cur, color="#40c4ff", lw=1.2)
        self.tl_fig.tight_layout()
        self.tl_canvas.draw_idle()

    def _on_timeline_click(self, event):
        if event.inaxes is self.tl_ax and event.xdata is not None:
            self.goto_frame(event.xdata)
            self.nb.select(0)

    def update_charts(self, timeline_only=False):
        """Redraw whichever chart tab is showing (charts on hidden tabs wait until shown)."""
        if not self.ev or Figure is None:
            return
        tab = self.nb.index("current")
        if tab == 2 and not timeline_only and self.curve_fig is not None:
            self._draw_curves()
        elif tab == 3 and self.tl_fig is not None:
            self._draw_timeline()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="data folder (can also be picked in the GUI)")
    ap.add_argument("--db", type=Path, help="detections.db from results_db.py")
    ap.add_argument("--json", type=Path, nargs="*", default=[],
                    help="small detection JSON file(s) (can also be added in the GUI)")
    ap.add_argument("--eval", type=Path, help="evaluate.py output folder (default: <db folder>/eval)")
    args = ap.parse_args()
    Viewer(args.data, args.json, args.db, args.eval).mainloop()


if __name__ == "__main__":
    main()
