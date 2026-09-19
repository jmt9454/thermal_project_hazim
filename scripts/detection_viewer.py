"""
Detection viewer for the episode/modality image layout:

    <data>/ep{N}/{rgb,thermal,rf}/frame_{K}_*.{jpg|png}

Load one or more detection JSONs (from detect_poses.py) and overlay their person
boxes, confidences, body-part boxes and keypoint skeletons on each frame.
Every (JSON file, model) pair is a "source" with its own colour that can be
toggled on/off.  Pick which modalities to show (1/2/3); the grid re-lays itself
out so the visible images are as large as possible.  One global threshold slider
hides everything (detections, parts, keypoints) scoring below it.
If an annotations.json from annotator.py sits next to or inside the data folder,
its human / no-human label is shown as ground truth.

Usage:
    python detection_viewer.py
    python detection_viewer.py --data ../data --json a.json b.json

Keys:
    <-/->  or  Up/Down     prev / next frame
    Home / End             first / last frame
    PageUp / PageDown      prev / next episode
    1 / 2 / 3              show / hide rgb / thermal / rf
    B  C  P  K             toggle boxes / conf labels / part boxes / keypoints
    [  ]                   threshold down / up (1% of the slider range)
    Esc                    quit
"""

import argparse
import json
import math
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk
except ImportError:
    sys.exit("Missing dependency: Pillow.  Install with:  pip install pillow")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FRAME_RE = re.compile(r"frame[_-]?(\d+)", re.I)
MOD_ORDER = {"rgb": 0, "thermal": 1, "rf": 2}

BG = "#1e1e1e"
PANEL_BG = "#252526"
CANVAS_BG = "#111"
COL_HUMAN = "#69f0ae"
COL_NO = "#ff5252"
COL_UNSET = "#888888"
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
# data helpers (the scanning ones are the same as annotator.py)
# ----------------------------------------------------------------------------
def nat_key(name):
    m = re.search(r"(\d+)\s*$", name)
    return (int(m.group(1)) if m else 1 << 30, name)


def mod_key(m):
    return (MOD_ORDER.get(m.lower(), 9), m.lower())


def scan_episodes(data_root):
    """-> {ep_name: {"frames": [int], "images": {modality: {frame_idx: Path}}}}"""
    episodes = {}
    for ep_dir in sorted(data_root.iterdir(), key=lambda p: nat_key(p.name)):
        if not ep_dir.is_dir():
            continue
        images, frames = {}, set()
        for mod_dir in ep_dir.iterdir():
            if not mod_dir.is_dir():
                continue
            for p in mod_dir.iterdir():
                m = FRAME_RE.search(p.stem)
                if p.suffix.lower() in IMG_EXTS and m:
                    images.setdefault(mod_dir.name.lower(), {})[int(m.group(1))] = p
                    frames.add(int(m.group(1)))
        if frames:
            episodes[ep_dir.name] = {"frames": sorted(frames), "images": images}
    return episodes


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


class Source:
    """One model's detections from one JSON file."""

    def __init__(self, json_path, json_no, data, model, color):
        self.tag = f"J{json_no}:{model}"          # short name used in tile titles
        self.file = json_path.name
        self.data, self.model, self.color = data, model, color
        self.var = tk.BooleanVar(value=True)  # visible?
        self.max = max((d["conf"] for e in data.values() for f in e.values()
                        for m in f.values() for d in m.get(model, [])), default=0.0)

    def dets(self, ep, frame, mod):
        return self.data.get(ep, {}).get(str(frame), {}).get(mod, {}).get(self.model, [])


# ----------------------------------------------------------------------------
# app
# ----------------------------------------------------------------------------
class Viewer(tk.Tk):
    def __init__(self, data_dir, json_paths):
        super().__init__()
        self.title("thermal_hazim detection viewer")
        self.geometry("1400x900")
        self.configure(bg=BG)

        self.episodes, self.ep_names, self.modalities = {}, [], []
        self.ann = {}          # ground truth from annotations.json
        self.sources = []      # list[Source]
        self.frame_idx = 0     # position within the current episode's frames
        self._photos, self._cache, self._pending = [], {}, None

        self.data_var = tk.StringVar()
        self.ep = tk.StringVar()
        self.thr = tk.DoubleVar(value=0.0)
        self.full_range = tk.BooleanVar(value=False)
        self.mod_vars = {}     # modality -> BooleanVar (visible in grid?)
        self.show = {k: tk.BooleanVar(value=True) for k in ("boxes", "conf", "parts", "kpts")}
        self.font = get_font(12)

        self._build_ui()
        self._bind_keys()
        if data_dir:
            self.set_data_dir(data_dir)
        for p in json_paths:
            self.add_json(p)

    # ---------------- UI ----------------
    def _build_ui(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        sep = lambda parent: ttk.Separator(parent, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8)

        # row 1: data folder, JSON files, episode
        top = ttk.Frame(self, padding=(6, 5))
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="Data:").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.data_var, width=45,
                  state="readonly").pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Browse…", command=self.browse_data).pack(side=tk.LEFT)
        sep(top)
        ttk.Button(top, text="Add JSON…", command=self.browse_json).pack(side=tk.LEFT)
        ttk.Button(top, text="Clear JSONs", command=self.clear_json).pack(side=tk.LEFT, padx=4)
        sep(top)
        ttk.Label(top, text="Episode:").pack(side=tk.LEFT)
        self.ep_menu = ttk.OptionMenu(top, self.ep, "")
        self.ep_menu.pack(side=tk.LEFT, padx=2)

        # row 2: which modalities are in the grid, what to draw, ground truth
        row2 = ttk.Frame(self, padding=(6, 2))
        row2.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(row2, text="Show:").pack(side=tk.LEFT)
        self.mod_bar = ttk.Frame(row2)  # filled in set_data_dir()
        self.mod_bar.pack(side=tk.LEFT)
        sep(row2)
        ttk.Label(row2, text="Draw:").pack(side=tk.LEFT)
        for key, text in (("boxes", "boxes [B]"), ("conf", "conf [C]"),
                          ("parts", "parts [P]"), ("kpts", "keypoints [K]")):
            ttk.Checkbutton(row2, text=text, variable=self.show[key],
                            command=self.render).pack(side=tk.LEFT, padx=2)
        self.gt_label = tk.Label(row2, text="", bg=BG, font=("Segoe UI", 12, "bold"))
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
                               command=lambda _v: self.schedule_render())
        self.scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        self.status = ttk.Label(
            self, anchor="w", padding=(6, 2),
            text="←/→ frame   PgUp/PgDn episode   1/2/3 modality   "
                 "B/C/P/K draw toggles   [ ] threshold   Esc quit")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        # main area: canvas | side panel (sources + frame list)
        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(body, bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _e: self.schedule_render())

        side = ttk.Frame(body, width=280, padding=(6, 4))
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)
        ttk.Label(side, text="Sources (J#:model)").pack(anchor="w")
        self.src_frame = tk.Frame(side, bg=PANEL_BG)
        self.src_frame.pack(fill=tk.X, pady=(2, 8))
        ttk.Label(side, text="Frames   (max conf, visible sources)").pack(anchor="w")
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

    def _bind_keys(self):
        self.bind("<Left>", lambda e: self.step(-1))
        self.bind("<Right>", lambda e: self.step(1))
        self.bind("<Up>", lambda e: self.step(-1))
        self.bind("<Down>", lambda e: self.step(1))
        self.bind("<Home>", lambda e: self.goto(0))
        self.bind("<End>", lambda e: self.goto(len(self._frames()) - 1))
        self.bind("<Prior>", lambda e: self.step_episode(-1))  # PageUp
        self.bind("<Next>", lambda e: self.step_episode(1))    # PageDown
        for i in range(1, 10):
            self.bind(str(i), lambda e, i=i: self.toggle_mod(i - 1))
        for key, name in (("b", "boxes"), ("c", "conf"), ("p", "parts"), ("k", "kpts")):
            self.bind(key, lambda e, n=name: self.toggle_draw(n))
            self.bind(key.upper(), lambda e, n=name: self.toggle_draw(n))
        self.bind("[", lambda e: self.nudge(-1))
        self.bind("]", lambda e: self.nudge(1))
        self.bind("<Escape>", lambda e: self.destroy())

    # ---------------- loading ----------------
    def browse_data(self):
        p = filedialog.askdirectory(title="Data folder (contains ep1, ep2, ...)",
                                    initialdir=self.data_var.get() or ".")
        if p:
            self.set_data_dir(p)

    def browse_json(self):
        start = Path(self.data_var.get()).parent if self.data_var.get() else "."
        for p in filedialog.askopenfilenames(title="Detection JSON file(s)", initialdir=start,
                                             filetypes=[("JSON", "*.json"), ("All", "*.*")]):
            self.add_json(p)

    def set_data_dir(self, path):
        path = Path(path).resolve()
        episodes = scan_episodes(path) if path.is_dir() else {}
        if not episodes:
            self.status.configure(text=f"no episodes with frame images under {path}")
            return
        self.data_var.set(str(path))
        self.episodes, self.ep_names = episodes, list(episodes)
        self.modalities = sorted({m for e in episodes.values() for m in e["images"]}, key=mod_key)
        self._cache.clear()

        # modality checkboxes: all visible to start
        for w in self.mod_bar.winfo_children():
            w.destroy()
        self.mod_vars = {m: tk.BooleanVar(value=True) for m in self.modalities}
        for i, m in enumerate(self.modalities, start=1):
            ttk.Checkbutton(self.mod_bar, text=f"{m.upper()} [{i}]", variable=self.mod_vars[m],
                            command=self.refresh).pack(side=tk.LEFT, padx=2)

        # episode drop-down
        menu = self.ep_menu["menu"]
        menu.delete(0, tk.END)
        for name in self.ep_names:
            menu.add_command(label=name, command=lambda n=name: (self.ep.set(n), self.load_episode()))
        self.ep.set(self.ep_names[0])

        # ground truth from annotator.py (it saves next to the data folder by default)
        self.ann = {}
        for p in (path.parent / "annotations.json", path / "annotations.json"):
            if p.is_file():
                self.ann = load_annotations(p)
                break
        self.load_episode()

    def add_json(self, path):
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            self.status.configure(text=f"could not read {path.name}: {e}")
            return
        json_no = len({id(s.data) for s in self.sources}) + 1  # J1, J2, ... per file
        for model in models_in(data):  # one source per model inside the file
            color = PALETTE[len(self.sources) % len(PALETTE)]
            self.sources.append(Source(path, json_no, data, model, color))
        self._rebuild_sources()

    def clear_json(self):
        self.sources = []
        self._rebuild_sources()

    def _rebuild_sources(self):
        """Redraw the coloured source checkboxes, slider range and frame list."""
        for w in self.src_frame.winfo_children():
            w.destroy()
        for s in self.sources:
            tk.Checkbutton(self.src_frame, text=f"{s.tag}   max {s.max:.3f}\n{s.file}",
                           variable=s.var, command=self.refresh, anchor="w",
                           justify="left", wraplength=250, fg=s.color, bg=PANEL_BG,
                           selectcolor=PANEL_BG, activebackground="#333",
                           activeforeground=s.color, font=("Consolas", 9)).pack(fill=tk.X)
        self.update_range()
        self.refresh()

    def update_range(self):
        """Slider spans 0..(highest conf in the loaded JSONs), or 0..1 if ticked."""
        data_max = max((s.max for s in self.sources), default=1.0)
        top = 1.0 if self.full_range.get() else max(data_max, 1e-3)
        self.scale.configure(to=top)
        if self.thr.get() > top:
            self.thr.set(top)
        self.range_label.configure(text=f"range 0–{top:.3f}")
        self.render()

    # ---------------- helpers ----------------
    def _frames(self):
        return self.episodes[self.ep.get()]["frames"] if self.episodes else []

    def _cur_frame(self):
        fr = self._frames()
        return fr[self.frame_idx] if fr else None

    def _image_path(self, modality, frame):
        return self.episodes[self.ep.get()]["images"].get(modality, {}).get(frame)

    def _visible_mods(self):
        return [m for m in self.modalities if self.mod_vars[m].get()]

    def _active_sources(self):
        return [s for s in self.sources if s.var.get()]

    def _gt(self, frame):
        return self.ann.get(self.ep.get(), {}).get(frame)  # True / False / None

    def _load(self, path):
        if path not in self._cache:
            if len(self._cache) > 24:
                self._cache.clear()
            self._cache[path] = Image.open(path).convert("RGB")
        return self._cache[path]

    # ---------------- navigation / toggles ----------------
    def load_episode(self):
        self.frame_idx = 0
        self.refresh()

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

    def step_episode(self, d):
        if self.ep_names:
            i = (self.ep_names.index(self.ep.get()) + d) % len(self.ep_names)
            self.ep.set(self.ep_names[i])
            self.load_episode()

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
        self.thr.set(min(top, max(0.0, self.thr.get() + d * top / 100)))
        self.render()

    def refresh(self):
        """Something changed that affects the frame list as well as the image."""
        self._fill_list()
        self.render()

    # ---------------- frame list ----------------
    def _fill_list(self):
        """One row per frame: ground-truth tag and the max conf of the visible sources."""
        self.listbox.delete(0, tk.END)
        ep, mods, srcs = self.ep.get(), self._visible_mods(), self._active_sources()
        for i, f in enumerate(self._frames()):
            best = max((d["conf"] for s in srcs for m in mods for d in s.dets(ep, f, m)),
                       default=None)
            gt = self._gt(f)
            tag, color = {True: ("H", COL_HUMAN), False: ("-", COL_NO)}.get(gt, ("?", COL_UNSET))
            text = f" {tag}  frame {f:<5}" + (f" max {best:.3f}" if best is not None else "")
            self.listbox.insert(tk.END, text)
            self.listbox.itemconfig(i, foreground=color)
        self._select_list_row()

    def _select_list_row(self):
        self.listbox.selection_clear(0, tk.END)
        if self._frames():
            self.listbox.selection_set(self.frame_idx)
            self.listbox.see(self.frame_idx)

    # ---------------- rendering ----------------
    def schedule_render(self):
        """Debounce: slider drags / window resizes redraw at most every 30 ms."""
        if self._pending:
            self.after_cancel(self._pending)
        self._pending = self.after(30, self.render)

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
                          text="Pick a data folder (Browse…) and add detection JSONs.")
            return
        self._refresh_header(frame)
        mods = self._visible_mods()
        if not mods:
            c.create_text(cw // 2, ch // 2, fill="#aaa", text="No modality shown — press 1/2/3")
            return

        ep, t = self.ep.get(), self.thr.get()
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
                for src in self._active_sources():
                    n = self._draw_dets(img, s, src.dets(ep, frame, m), src.color, t)
                    title.append((f"{src.tag}: {n}", src.color))
                self._photos.append(ImageTk.PhotoImage(img))
                c.create_image(x0 + tw / 2, y0 + HEADER + (th - HEADER) / 2,
                               image=self._photos[-1], anchor="center")
            # tile title: modality, then "source: #shown" in each source's colour
            x = x0 + 6
            for text, color in title:
                if x > x0 + tw - 40:  # out of room in this tile
                    break
                item = c.create_text(x, y0 + 4, anchor="nw", text=text, fill=color,
                                     font=("Segoe UI", 10, "bold"))
                x = c.bbox(item)[2] + 12

    def _draw_dets(self, img, s, dets, color, t):
        """Draw every detection with conf >= t onto img (image scale s).
        Parts and keypoints are also filtered by their own conf >= t.
        Returns how many detections were drawn."""
        d = ImageDraw.Draw(img)
        sc = lambda xs: [v * s for v in xs]
        shown = 0
        for det in sorted(dets, key=lambda x: x["conf"]):  # weakest first, strongest on top
            if det["conf"] < t:
                continue
            shown += 1
            if self.show["kpts"].get() and "keypoints" in det:
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
            if self.show["parts"].get():
                for part in det.get("parts", {}).values():
                    if part["conf"] >= t:
                        d.rectangle(sc(part["box"]), outline=color, width=1)
            if self.show["boxes"].get():
                d.rectangle(sc(det["box"]), outline=color, width=2)
            if self.show["conf"].get():
                x, y = sc(det["box"][:2])
                label = f"{det['conf']:.3f}"
                h = d.textbbox((0, 0), label, font=self.font)[3] + 2
                y = max(0, y - h)  # just above the box
                d.rectangle(d.textbbox((x + 1, y), label, font=self.font), fill="black")
                d.text((x + 1, y), label, fill=color, font=self.font)
        return shown

    def _refresh_header(self, frame):
        gt = self._gt(frame)
        text = {True: "GT: HUMAN", False: "GT: NO HUMAN"}.get(gt, "GT: —" if self.ann else "")
        color = {True: COL_HUMAN, False: COL_NO}.get(gt, COL_UNSET)
        self.gt_label.configure(text=text, fg=color)
        pos = f"{self.frame_idx + 1}/{len(self._frames())}"
        self.title(f"thermal_hazim detection viewer  [{self.ep.get()}  frame {frame}  {pos}]")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="episode folder (can also be picked in the GUI)")
    ap.add_argument("--json", type=Path, nargs="*", default=[],
                    help="detection JSON file(s) (can also be added in the GUI)")
    args = ap.parse_args()
    Viewer(args.data, args.json).mainloop()


if __name__ == "__main__":
    main()