"""
Quick human-presence annotator for the episode/modality image layout:

    data/ep{N}/{rgb,thermal,rf}/frame_{K}_{modality}.{jpg|png}

Pick an episode, flip between modalities (they are the same moment seen three
ways), and toggle whether a human is present in each frame.  Human presence is a
per-frame label shared across modalities.  Save writes a single annotations.json
at the project root and reloads it on the next run, so labelling is resumable.

Usage:
    python annotator.py
    python annotator.py --data ".\\data" --out ".\\annotations.json"

Keys:
    <-/->  or  Up/Down     prev / next frame
    Home / End             first / last frame
    PageUp / PageDown      prev / next episode
    1 / 2 / 3              rgb / thermal / rf
    Space or Y            mark HUMAN present
    N                     mark NO human
    U                     clear (back to un-reviewed)
    Ctrl+S or S           save annotations.json
    Esc                   quit (auto-saves if there are unsaved changes)
"""

import argparse
import json
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

try:
    from PIL import Image, ImageTk
except ImportError:
    sys.exit("Missing dependency: Pillow.  Install with:  pip install pillow")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FRAME_RE = re.compile(r"frame[_-]?(\d+)", re.I)
MOD_ORDER = {"rgb": 0, "thermal": 1, "rf": 2}

BG = "#1e1e1e"
CANVAS_BG = "#111"
COL_HUMAN = "#69f0ae"
COL_NO = "#ff5252"
COL_UNSET = "#888888"


# ----------------------------------------------------------------------------
# data
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
            mod = mod_dir.name.lower()
            for p in mod_dir.iterdir():
                if p.suffix.lower() not in IMG_EXTS:
                    continue
                m = FRAME_RE.search(p.stem)
                if not m:
                    continue
                idx = int(m.group(1))
                images.setdefault(mod, {})[idx] = p
                frames.add(idx)
        if frames:
            episodes[ep_dir.name] = {"frames": sorted(frames), "images": images}
    return episodes


def fit(img, box_w, box_h):
    """Return (resized_image, scale) so img fits inside box_w x box_h."""
    if box_w <= 1 or box_h <= 1:
        return img, 1.0
    s = min(box_w / img.width, box_h / img.height)
    new = (max(1, int(img.width * s)), max(1, int(img.height * s)))
    return img.resize(new, Image.BILINEAR), s


# ----------------------------------------------------------------------------
# app
# ----------------------------------------------------------------------------
class Annotator(tk.Tk):
    def __init__(self, episodes, modalities, out_path, ann):
        super().__init__()
        self.title("thermal_hazim annotator")
        self.geometry("1200x820")
        self.configure(bg=BG)

        self.episodes = episodes
        self.ep_names = list(episodes.keys())
        self.modalities = modalities
        self.out_path = out_path
        # ann: {ep_name: {frame_idx(int): bool}}
        self.ann = ann
        self.dirty = False

        self.ep = tk.StringVar(value=self.ep_names[0])
        self.modality = tk.StringVar(value=modalities[0])
        self.frame_idx = 0                 # position within current episode's frames
        self._photo = None
        self._cache = {}

        self._build_ui()
        self._bind_keys()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.load_episode()

    # ---------------- UI ----------------
    def _build_ui(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        top = ttk.Frame(self, padding=(6, 5))
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Episode:").pack(side=tk.LEFT)
        self.ep_menu = ttk.OptionMenu(top, self.ep, self.ep_names[0], *self.ep_names,
                                      command=lambda _v: self.load_episode())
        self.ep_menu.pack(side=tk.LEFT, padx=(2, 8))

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Label(top, text="Modality:").pack(side=tk.LEFT)
        for i, mod in enumerate(self.modalities, start=1):
            ttk.Radiobutton(top, text=f"{mod.upper()} [{i}]", value=mod,
                            variable=self.modality,
                            command=self.render).pack(side=tk.LEFT)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self.save_btn = ttk.Button(top, text="Save annotations.json  [Ctrl+S]",
                                   command=self.save)
        self.save_btn.pack(side=tk.LEFT)
        self.save_status = ttk.Label(top, text="")
        self.save_status.pack(side=tk.LEFT, padx=(8, 0))

        # human-presence controls
        pres = ttk.Frame(self, padding=(6, 4))
        pres.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(pres, text="Human present?").pack(side=tk.LEFT)
        ttk.Button(pres, text="HUMAN  [Y / Space]",
                   command=lambda: self.set_state(True)).pack(side=tk.LEFT, padx=4)
        ttk.Button(pres, text="NO human  [N]",
                   command=lambda: self.set_state(False)).pack(side=tk.LEFT, padx=4)
        ttk.Button(pres, text="clear  [U]",
                   command=lambda: self.set_state(None)).pack(side=tk.LEFT, padx=4)
        self.state_label = tk.Label(pres, text="", bg=BG, font=("Segoe UI", 13, "bold"))
        self.state_label.pack(side=tk.LEFT, padx=16)
        self.progress_label = ttk.Label(pres, text="")
        self.progress_label.pack(side=tk.RIGHT)

        # main area: canvas | frame list
        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(body, bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _e: self.render())

        side = ttk.Frame(body, width=240, padding=(6, 4))
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)
        ttk.Label(side, text="Frames (click to jump)").pack(anchor="w")
        lb_frame = ttk.Frame(side)
        lb_frame.pack(fill=tk.BOTH, expand=True)
        self.listbox = tk.Listbox(lb_frame, bg="#252526", fg="#ddd",
                                  font=("Consolas", 10), selectbackground="#0a84ff",
                                  activestyle="none", exportselection=False)
        sb = ttk.Scrollbar(lb_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)

        self.status = ttk.Label(
            self, anchor="w", padding=(6, 2),
            text="←/→ frame   PgUp/PgDn episode   1/2/3 modality   "
                 "Y/Space human   N no-human   U clear   Ctrl+S save   Esc quit")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_keys(self):
        self.bind("<Left>", lambda e: self.step(-1))
        self.bind("<Right>", lambda e: self.step(1))
        self.bind("<Up>", lambda e: self.step(-1))
        self.bind("<Down>", lambda e: self.step(1))
        self.bind("<Home>", lambda e: self.goto(0))
        self.bind("<End>", lambda e: self.goto(len(self._frames()) - 1))
        self.bind("<Prior>", lambda e: self.step_episode(-1))   # PageUp
        self.bind("<Next>", lambda e: self.step_episode(1))     # PageDown
        for i, mod in enumerate(self.modalities, start=1):
            if i <= 9:
                self.bind(str(i), lambda e, m=mod: self._set_modality(m))
        self.bind("<space>", lambda e: self.set_state(True))
        self.bind("y", lambda e: self.set_state(True))
        self.bind("Y", lambda e: self.set_state(True))
        self.bind("n", lambda e: self.set_state(False))
        self.bind("N", lambda e: self.set_state(False))
        self.bind("u", lambda e: self.set_state(None))
        self.bind("U", lambda e: self.set_state(None))
        self.bind("<Control-s>", lambda e: self.save())
        self.bind("s", lambda e: self.save())
        self.bind("S", lambda e: self.save())
        self.bind("<Escape>", lambda e: self._on_close())

    # ---------------- helpers ----------------
    def _frames(self):
        return self.episodes[self.ep.get()]["frames"]

    def _cur_frame(self):
        fr = self._frames()
        return fr[self.frame_idx] if fr else None

    def _image_path(self, modality, frame):
        return self.episodes[self.ep.get()]["images"].get(modality, {}).get(frame)

    def _get_ann(self, ep=None, frame=None):
        ep = ep or self.ep.get()
        frame = self._cur_frame() if frame is None else frame
        return self.ann.get(ep, {}).get(frame)   # True / False / None

    def _set_modality(self, m):
        self.modality.set(m)
        self.render()

    # ---------------- state changes ----------------
    def set_state(self, value):
        """value: True=human, False=no-human, None=clear."""
        frame = self._cur_frame()
        if frame is None:
            return
        ep_map = self.ann.setdefault(self.ep.get(), {})
        if value is None:
            ep_map.pop(frame, None)
        else:
            ep_map[frame] = bool(value)
        self._mark_dirty()
        self._refresh_state_label()
        self._refresh_list_item(self.frame_idx)
        self._refresh_progress()

    def _mark_dirty(self):
        self.dirty = True
        self.save_status.configure(text="unsaved changes", foreground=COL_NO)
        self._refresh_title()

    def load_episode(self):
        self.frame_idx = 0
        self._fill_list()
        self.render()

    def step(self, d):
        fr = self._frames()
        if fr:
            self.goto((self.frame_idx + d) % len(fr))

    def goto(self, i):
        fr = self._frames()
        if fr:
            self.frame_idx = max(0, min(i, len(fr) - 1))
            self.render()

    def step_episode(self, d):
        i = (self.ep_names.index(self.ep.get()) + d) % len(self.ep_names)
        self.ep.set(self.ep_names[i])
        self.load_episode()

    def _on_list_select(self, _e):
        sel = self.listbox.curselection()
        if sel and sel[0] != self.frame_idx:
            self.goto(sel[0])

    # ---------------- list panel ----------------
    def _tag_for(self, value):
        if value is True:
            return "H", COL_HUMAN
        if value is False:
            return "-", COL_NO
        return "?", COL_UNSET

    def _fill_list(self):
        self.listbox.delete(0, tk.END)
        for i, frame in enumerate(self._frames()):
            self.listbox.insert(tk.END, "")
            self._refresh_list_item(i)

    def _refresh_list_item(self, i):
        frame = self._frames()[i]
        tag, color = self._tag_for(self._get_ann(frame=frame))
        self.listbox.delete(i)
        self.listbox.insert(i, f" {tag}  frame {frame}")
        self.listbox.itemconfig(i, foreground=color)
        if i == self.frame_idx:
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(i)
            self.listbox.see(i)

    # ---------------- rendering ----------------
    def _load(self, path):
        if path not in self._cache:
            if len(self._cache) > 24:
                self._cache.clear()
            self._cache[path] = Image.open(path).convert("RGB")
        return self._cache[path]

    def render(self):
        self.canvas.delete("all")
        cw, ch = max(self.canvas.winfo_width(), 2), max(self.canvas.winfo_height(), 2)
        frame = self._cur_frame()
        if frame is None:
            self.canvas.create_text(20, 20, anchor="nw", fill="#aaa",
                                    text="No frames in this episode.")
            return

        path = self._image_path(self.modality.get(), frame)
        if path is None:
            self.canvas.create_text(
                cw // 2, ch // 2, fill="#aaa", justify="center",
                text=f"{self.modality.get()} view missing\nfor frame {frame}")
        else:
            img, _ = fit(self._load(path), cw, ch)
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor="center")

        self._refresh_state_label()
        self._refresh_progress()
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(self.frame_idx)
        self.listbox.see(self.frame_idx)
        self._refresh_title()

    def _refresh_state_label(self):
        value = self._get_ann()
        text = {True: "HUMAN PRESENT", False: "NO HUMAN", None: "— not reviewed —"}[value]
        _, color = self._tag_for(value)
        self.state_label.configure(text=text, fg=color)

    def _refresh_progress(self):
        frames = self._frames()
        done = sum(1 for f in frames if self._get_ann(frame=f) is not None)
        total_frames = sum(len(e["frames"]) for e in self.episodes.values())
        total_done = sum(1 for ep, e in self.episodes.items()
                         for f in e["frames"] if self.ann.get(ep, {}).get(f) is not None)
        self.progress_label.configure(
            text=f"episode {done}/{len(frames)}    total {total_done}/{total_frames}")

    def _refresh_title(self):
        frame = self._cur_frame()
        star = "*" if self.dirty else ""
        pos = f"{self.frame_idx + 1}/{len(self._frames())}" if self._frames() else "-"
        self.title(f"{star}thermal_hazim annotator  [{self.ep.get()}  frame {frame}  {pos}]")

    # ---------------- persistence ----------------
    def save(self):
        payload = {ep: {str(f): v for f, v in sorted(fr.items())}
                   for ep, fr in self.ann.items() if fr}
        self.out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.dirty = False
        self.save_status.configure(text=f"saved -> {self.out_path.name}",
                                   foreground=COL_HUMAN)
        self._refresh_title()

    def _on_close(self):
        if self.dirty:
            self.save()
        self.destroy()


# ----------------------------------------------------------------------------
def load_annotations(out_path):
    """Read annotations.json into {ep: {int frame: bool}} (empty if absent)."""
    if not out_path.is_file():
        return {}
    try:
        raw = json.loads(out_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    ann = {}
    for ep, frames in (raw or {}).items():
        if isinstance(frames, dict):
            ann[ep] = {int(k): bool(v) for k, v in frames.items()}
    return ann


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--data", type=Path, default=here / "data",
                    help="folder of episodes (default: <script>/data)")
    ap.add_argument("--out", type=Path, default=here / "annotations.json",
                    help="annotations JSON at project root (default: <script>/annotations.json)")
    args = ap.parse_args()

    data = args.data.resolve()
    if not data.is_dir():
        sys.exit(f"data folder not found: {data}")

    episodes = scan_episodes(data)
    if not episodes:
        sys.exit(f"no episodes with frame images found under {data}")

    modalities = sorted({m for e in episodes.values() for m in e["images"]}, key=mod_key)
    total = sum(len(e["frames"]) for e in episodes.values())
    print(f"{len(episodes)} episodes, {total} frames, modalities: {', '.join(modalities)}")

    ann = load_annotations(args.out.resolve())
    if ann:
        done = sum(1 for ep, e in episodes.items()
                   for f in e["frames"] if ann.get(ep, {}).get(f) is not None)
        print(f"loaded {done} existing labels from {args.out}")

    Annotator(episodes, modalities, args.out.resolve(), ann).mainloop()


if __name__ == "__main__":
    main()
