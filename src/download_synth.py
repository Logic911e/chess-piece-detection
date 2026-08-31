"""Download the chesscog rendered dataset from OSF.

    Woelflein, G. and Arandjelovic, O. "Dataset of Rendered Chess Game State
    Images." DOI 10.17605/OSF.IO/XF3KA

Roughly 4,888 rendered chess positions, about 2.5 GB.

This replaces chesscog's own downloader, which requires installing the whole
package (torch, opencv, recap) just to fetch files, clones into a temporary
directory, and offers no resume. A dropped connection there discards
everything downloaded so far.

Here every file is downloaded to a persistent `.part` file and resumed with an
HTTP Range request after an interruption. Nothing is deleted on failure, so
re-running the script continues where it stopped. Splits that have already
been extracted are skipped entirely.

Usage
-----
    python src/download_synth.py            # download, extract, verify
    python src/download_synth.py --skip-download   # extract what is present
    python src/download_synth.py --clean    # delete archives after extracting
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import config
import osf_download

OSF_PROJECT = "xf3ka"

# Which archives provide which split. The train split is a two-part archive.
SPLIT_ARCHIVES: dict[str, tuple[str, ...]] = {
    "val": ("val.zip",),
    "test": ("test.zip",),
    "train": ("train.zip", "train.z01"),
}


def split_is_extracted(dest: Path, split: str) -> bool:
    """True if the split directory already holds rendered images."""
    d = dest / split
    return d.is_dir() and any(d.glob("*.png"))


def extract_simple(dest: Path, archive: str) -> bool:
    path = dest / archive
    if not path.is_file():
        print(f"  [warn] {archive} not found")
        return False
    print(f"  [unzip] {archive}")
    with zipfile.ZipFile(path) as zf:
        zf.extractall(dest)
    return True


def extract_train(dest: Path) -> bool:
    """Join and extract the two-part train archive.

    Python's zipfile cannot read multi-part archives, so this shells out to
    the `zip` utility to join them first.
    """
    main = dest / "train.zip"
    part1 = dest / "train.z01"
    merged = dest / "train_full.zip"

    if not main.is_file():
        print("  [warn] train.zip not found")
        return False

    if not part1.is_file():
        print("  [unzip] train.zip (single archive)")
        with zipfile.ZipFile(main) as zf:
            zf.extractall(dest)
        return True

    if shutil.which("zip") is None:
        print(
            "\n  [error] The `zip` utility is needed to join the split train\n"
            "  archive but is not on PATH.\n"
            f"  Join manually in {dest}:\n"
            "      zip -s 0 train.zip --out train_full.zip\n"
            "      unzip train_full.zip"
        )
        return False

    print("  [zip] joining train.zip + train.z01 (slow, needs ~2.5 GB free)")
    result = subprocess.run(
        ["zip", "-s", "0", str(main), "--out", str(merged)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not merged.is_file():
        print(f"  [error] join failed: {result.stderr.strip()}")
        print(
            f"\n  Join manually in {dest}:\n"
            "      zip -s 0 train.zip --out train_full.zip\n"
            "      unzip train_full.zip"
        )
        return False

    print("  [unzip] train_full.zip")
    with zipfile.ZipFile(merged) as zf:
        zf.extractall(dest)
    merged.unlink(missing_ok=True)
    return True


def verify(dest: Path) -> int:
    """Report what was extracted and confirm per-piece boxes are present."""
    print(f"\n{'=' * 60}\nCONTENTS\n{'=' * 60}")
    total_png = total_json = 0
    for split in ("train", "val", "test"):
        d = dest / split
        if not d.is_dir():
            print(f"  {split:6} missing")
            continue
        n_png = len(list(d.glob("*.png")))
        n_json = len(list(d.glob("*.json")))
        total_png += n_png
        total_json += n_json
        note = "" if n_png == n_json else "   <-- counts differ"
        print(f"  {split:6} {n_png:5d} images, {n_json:5d} annotations{note}")
    print(f"  {'total':6} {total_png:5d} images, {total_json:5d} annotations")

    sample = next(dest.rglob("*.json"), None)
    if sample is None:
        print("\n  [warn] no annotations found")
        return 1

    data = json.loads(sample.read_text())
    print(f"\nSample annotation ({sample.name})")
    print(f"  keys: {sorted(data)}")

    pieces = data.get("pieces")
    if pieces and isinstance(pieces, list) and "box" in pieces[0]:
        print(f"  pieces[0]: {pieces[0]}")
        print("\n  Per-piece boxes present. Convert with:")
        print("      python src/synth_to_yolo.py \\")
        print(f"          --src {dest} \\")
        print("          --dest data/processed/synth --copy-images")
        return 0

    print("\n  [warn] no per-piece 'box' field in this annotation.")
    print("  Boxes will have to be derived from 'corners' and 'fen' instead.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest", type=Path, default=config.RAW_DIR / "render",
        help="Where to put the dataset (default: data/raw/render)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Only extract and verify files already on disk",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Delete the archives after extracting (saves ~2.5 GB)",
    )
    parser.add_argument(
        "--retries", type=int, default=osf_download.DEFAULT_RETRIES,
        help="Retry attempts per file before giving up",
    )
    args = parser.parse_args()

    config.ensure_dirs()
    dest = args.dest
    dest.mkdir(parents=True, exist_ok=True)

    already = [s for s in SPLIT_ARCHIVES if split_is_extracted(dest, s)]
    if already:
        print(f"[skip] already extracted: {', '.join(sorted(already))}")

    needed = {
        name
        for split, archives in SPLIT_ARCHIVES.items()
        if split not in already
        for name in archives
    }

    if not needed:
        print("[ok] every split is already extracted")
        return verify(dest)

    if not args.skip_download:
        try:
            files = osf_download.list_files(OSF_PROJECT)
        except Exception as exc:
            print(f"[error] could not list OSF project {OSF_PROJECT}: {exc}")
            return 1

        wanted = [f for f in files if f.name in needed]
        if not wanted:
            print(f"[warn] none of {sorted(needed)} found in the OSF project.")
            print(f"       available: {[f.name for f in files]}")
            return 1

        total = sum(f.size for f in wanted)
        print(f"\n[osf] need {len(wanted)} file(s), {osf_download.human(total)}")
        print("      interrupted downloads resume, so re-running is safe\n")

        for f in wanted:
            try:
                osf_download.download_file(f, dest, retries=args.retries)
            except Exception as exc:
                print(f"\n[error] {exc}")
                print("Re-run the same command to resume.")
                return 1

    print(f"\n{'=' * 60}\nEXTRACTING\n{'=' * 60}")
    for split in ("val", "test", "train"):
        if split in already:
            continue
        print(f"\n[{split}]")
        ok = extract_train(dest) if split == "train" else extract_simple(
            dest, f"{split}.zip"
        )
        if ok and args.clean:
            for name in SPLIT_ARCHIVES[split]:
                (dest / name).unlink(missing_ok=True)
            print(f"  [clean] removed {split} archive(s)")

    return verify(dest)


if __name__ == "__main__":
    raise SystemExit(main())
