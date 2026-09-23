"""
Walks data/EP_*/Traj_*/{IR,RGB}/ (any depth works) and runs the RF in-frame script in each folder.
 
The per-folder script processes every *.json in its current working directory,
so this runner just sets cwd to each IR/RGB folder and calls it there.
 
Usage:
    python run_all_episodes.py                          # defaults below
    python run_all_episodes.py --data data --script add_rf_frame.py
    python run_all_episodes.py --list                   # only enumerate, don't process
    python run_all_episodes.py --modalities IR          # only IR folders
"""
import argparse
import subprocess
import sys
from pathlib import Path
 
 
def find_frame_dirs(data_root: Path, modalities):
    """Yield (episode_dir, modality_dir, json_files) for every folder named
    IR/RGB (case-insensitive) anywhere under data_root. The parent of that
    folder is treated as the episode, so data/EP_*/IR, data/x/EP_*/IR, etc.
    all work."""
    wanted = {m.lower() for m in modalities}
    mod_dirs = sorted(
        d for d in data_root.rglob("*")
        if d.is_dir() and d.name.lower() in wanted
    )
    for mod_dir in mod_dirs:
        json_files = sorted(
            f for f in mod_dir.iterdir()
            if f.is_file()
            and f.suffix.lower() == ".json"
            and not f.stem.endswith("_with_rf_frame")
        )
        yield mod_dir.parent, mod_dir, json_files
 
 
def print_layout_hint(data_root: Path, max_entries=15):
    """Show what's actually under data_root when nothing matched."""
    print("Nothing matched. Top of the data folder looks like:")
    entries = sorted(data_root.iterdir())
    for e in entries[:max_entries]:
        print(f"  {e.name}{'/' if e.is_dir() else ''}")
        if e.is_dir():
            for sub in sorted(e.iterdir())[:5]:
                print(f"      {sub.name}{'/' if sub.is_dir() else ''}")
    if len(entries) > max_entries:
        print(f"  ... ({len(entries) - max_entries} more)")
 
 
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data", help="Root data folder (default: data)")
    parser.add_argument("--script", default="rf_mod.py",
                        help="Path to the per-folder RF in-frame script")
    parser.add_argument("--modalities", nargs="+", default=["IR", "RGB"],
                        help="Subfolders to process (default: IR RGB)")
    parser.add_argument("--list", action="store_true",
                        help="Only list the folders/JSON files found, do not process")
    args = parser.parse_args()
 
    data_root = Path(args.data).resolve()
    script = Path(args.script).resolve()
 
    if not data_root.is_dir():
        sys.exit(f"Data folder not found: {data_root}")
    if not args.list and not script.is_file():
        sys.exit(f"Script not found: {script}")
 
    dirs = list(find_frame_dirs(data_root, args.modalities))
    total_json = sum(len(files) for _, _, files in dirs)
    # Top-level folder under data (EP_*) counts as the episode
    episodes = {mod.relative_to(data_root).parts[0] for _, mod, _ in dirs}
 
    print(f"Data root: {data_root}")
    print(f"Found {len(episodes)} episodes, {len(dirs)} folders, "
          f"{total_json} input JSON files\n")
 
    if not dirs:
        print_layout_hint(data_root)
        return
 
    if args.list:
        for ep_dir, mod_dir, files in dirs:
            print(f"{mod_dir.relative_to(data_root).as_posix()}: {len(files)} json")
        return
 
    ok, failed = 0, []
    for ep_dir, mod_dir, files in dirs:
        label = mod_dir.relative_to(data_root).as_posix()
        if not files:
            print(f"[skip] {label}: no JSON files")
            continue
 
        print(f"\n########## {label} ({len(files)} json) ##########")
        result = subprocess.run([sys.executable, str(script)], cwd=mod_dir)
        if result.returncode == 0:
            ok += 1
        else:
            failed.append(label)
            print(f"[fail] {label}: exit code {result.returncode}")
 
    print("\n===================================")
    print("All episodes finished")
    print(f"Folders processed: {ok}")
    print(f"Folders failed:    {len(failed)}")
    for label in failed:
        print(f"    {label}")
    print("===================================")
 
 
if __name__ == "__main__":
    main()