"""Download and inspect the Chess Recognition Dataset (ChessReD).

    Masouris, A. and van Gemert, J. "End-to-End Chess Recognition."
    VISAPP 2024. DOI 10.5220/0012370200003660
    Dataset: DOI 10.4121/99b5c721-280b-450b-b058-b2900b69a90f.v2
    Licence: CC BY-NC-SA 4.0 (non-commercial, share-alike)

10,800 real smartphone photographs across 100 chess games, captured on three
different handsets (iPhone 12, Huawei P40 Pro, Samsung Galaxy S8) from angles
ranging from top-down to oblique, under both natural and artificial light.

A 2,078-image subset called ChessReD2K additionally carries per-piece bounding
boxes and chessboard corner annotations:

    train 1,442   val 330   test 306

That 306-image real test set is almost exactly the size of the 342-image
synthetic test set, which makes the synthetic-to-real comparison unusually
clean.

The annotations file is small; the images archive is very large. This script
therefore fetches and inspects the annotations first, reports exactly what the
bounding box fields look like, and only downloads images when asked.

Usage
-----
    python src/download_chessred.py                 # annotations + inspect
    python src/download_chessred.py --images        # also fetch the images
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import requests

import config
import osf_download

BASE = "https://data.4tu.nl/file/99b5c721-280b-450b-b058-b2900b69a90f"
ANNOTATIONS_URL = f"{BASE}/3cae6364-daca-4967-b426-1e4b68cdb64c"

# Two image archives are published. Only the 2K subset carries bounding boxes,
# and it is a fifth of the size, so it is the default here.
#
#   chessred2k.zip    4.68 GB   2,078 images, all box-annotated
#   chessred.zip     24.57 GB   10,800 images, 8,722 of them unannotated
CHESSRED2K_URL = f"{BASE}/410e41c7-dde5-413f-8722-3e112363a1a2"
CHESSRED_FULL_URL = f"{BASE}/6329e969-616e-48e3-b893-a0379d1c15ba"

ANNOTATIONS_MD5 = "d34bca5ad46ec7a8df96a1d3c36784f3"
CHESSRED_FULL_MD5 = "32e23ed535239dc517a6499762d9847e"


def remote_size(url: str, timeout: int = 30) -> int:
    """Content-Length for a URL, or 0 if the server will not say."""
    try:
        resp = requests.head(url, allow_redirects=True, timeout=timeout)
        return int(resp.headers.get("Content-Length") or 0)
    except (requests.RequestException, ValueError):
        return 0


def fetch(url: str, name: str, dest: Path, retries: int = 6) -> Path:
    """Download one file with resume support, reusing the OSF downloader."""
    size = remote_size(url)
    if size:
        print(f"[size] {name}: {osf_download.human(size)}")
    return osf_download.download_file(
        osf_download.OSFFile(name=name, size=size, download_url=url),
        dest,
        retries=retries,
    )


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def describe(value: Any, depth: int = 0) -> str:
    """One-line description of a JSON value's shape."""
    if isinstance(value, dict):
        return f"dict with {len(value)} keys: {sorted(value)[:8]}"
    if isinstance(value, list):
        if not value:
            return "empty list"
        first = value[0]
        if isinstance(first, dict):
            return f"list[{len(value)}] of dict, keys: {sorted(first)}"
        return f"list[{len(value)}] of {type(first).__name__}"
    return f"{type(value).__name__}: {value!r}"


def inspect(path: Path) -> int:
    """Report the annotation structure in the detail needed to write a converter."""
    print(f"\n{'=' * 68}\nANNOTATION STRUCTURE\n{'=' * 68}")
    with path.open() as fh:
        data = json.load(fh)

    print(f"\ntop-level keys: {sorted(data)}\n")
    for key in sorted(data):
        print(f"  {key:16} {describe(data[key])}")

    # Categories: the id -> piece-name mapping the converter needs.
    cats = data.get("categories")
    if isinstance(cats, list) and cats:
        print(f"\n{'-' * 68}\nCATEGORIES\n{'-' * 68}")
        for c in cats:
            print(f"  {c.get('id'):>3}  {c.get('name')}")

    # Annotations may be a dict of several annotation types.
    anns = data.get("annotations")
    if isinstance(anns, dict):
        print(f"\n{'-' * 68}\nANNOTATION GROUPS\n{'-' * 68}")
        for key, value in anns.items():
            print(f"  {key:16} {describe(value)}")

        pieces = anns.get("pieces")
        if isinstance(pieces, list) and pieces:
            print(f"\n{'-' * 68}\nPIECE ANNOTATIONS\n{'-' * 68}")
            print(f"  total: {len(pieces)}")
            print(f"  fields: {sorted(pieces[0])}")
            print(f"\n  first record:")
            for k, v in sorted(pieces[0].items()):
                print(f"      {k:22} {v!r}")

            # The crucial question: how many carry a bounding box, and what
            # does that box look like?
            box_field = None
            for candidate in ("bbox", "box", "bounding_box"):
                if any(candidate in p for p in pieces[:2000]):
                    box_field = candidate
                    break

            if box_field:
                with_box = [p for p in pieces if p.get(box_field) is not None]
                print(f"\n  BOUNDING BOXES FOUND in field '{box_field}'")
                print(f"      annotations with a box: {len(with_box)} "
                      f"of {len(pieces)}")
                if with_box:
                    print(f"      example: {with_box[0][box_field]}")
                    imgs = {p.get("image_id") for p in with_box}
                    print(f"      distinct images with boxes: {len(imgs)}")
                    counts = Counter(p.get("category_id") for p in with_box)
                    print(f"      distinct categories: {len(counts)}")
            else:
                print("\n  NO bounding box field found on piece annotations.")
                print("  Boxes may live in a separate ChessReD2K download.")

    elif isinstance(anns, list) and anns:
        print(f"\n{'-' * 68}\nANNOTATIONS (flat list)\n{'-' * 68}")
        print(f"  total: {len(anns)}")
        print(f"  fields: {sorted(anns[0])}")
        print(f"  first record: {anns[0]}")

    imgs = data.get("images")
    if isinstance(imgs, list) and imgs:
        print(f"\n{'-' * 68}\nIMAGES\n{'-' * 68}")
        print(f"  total: {len(imgs)}")
        print(f"  fields: {sorted(imgs[0])}")
        print(f"  first record: {imgs[0]}")

    splits = data.get("splits")
    if splits:
        print(f"\n{'-' * 68}\nSPLITS\n{'-' * 68}")
        print(f"  {describe(splits)}")
        if isinstance(splits, dict):
            for k, v in splits.items():
                print(f"      {k:16} {describe(v)}")

    print(f"\n{'=' * 68}")
    print("Send this output back before converting. The converter needs the")
    print("exact box field name, its coordinate convention, and the category")
    print("id to piece mapping.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path,
                        default=config.RAW_DIR / "chessred")
    parser.add_argument("--images", action="store_true",
                        help="Also download images (ChessReD2K, 4.68 GB)")
    parser.add_argument("--full", action="store_true",
                        help="Fetch all 10,800 images (24.57 GB) instead of "
                             "the 2,078 box-annotated ones. Rarely needed.")
    parser.add_argument("--extract", action="store_true",
                        help="Extract the images archive after downloading")
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()

    config.ensure_dirs()
    args.dest.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("ChessReD  (Masouris & van Gemert, VISAPP 2024)")
    print("Licence: CC BY-NC-SA 4.0 -- non-commercial use only")
    print("=" * 68)

    ann_path = args.dest / "annotations.json"
    if ann_path.is_file():
        print(f"\n[have] {ann_path.name} "
              f"({osf_download.human(ann_path.stat().st_size)})")
    else:
        print("\n[1/2] annotations")
        try:
            fetch(ANNOTATIONS_URL, "annotations.json", args.dest, args.retries)
        except Exception as exc:
            print(f"[error] {exc}")
            return 1

    if args.images:
        url = CHESSRED_FULL_URL if args.full else CHESSRED2K_URL
        zip_name = "chessred.zip" if args.full else "chessred2k.zip"
        size = remote_size(url)

        print(f"\n[2/2] {zip_name}: "
              f"{osf_download.human(size) if size else 'unknown size'}")
        if args.full:
            print("      All 10,800 images. Note that 8,722 of them carry no")
            print("      bounding boxes and cannot be used for detection.")
        else:
            print("      The 2,078 box-annotated ChessReD2K images.")
            print("      Use --full only if you need the unannotated remainder.")
        print("      Interrupted downloads resume, so re-running is safe.\n")

        try:
            zip_path = fetch(url, zip_name, args.dest, args.retries)
        except Exception as exc:
            print(f"[error] {exc}")
            print("Re-run to resume.")
            return 1

        if args.extract:
            print(f"\n[unzip] {zip_path.name}")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(args.dest)
            print("[done] extracted")
    else:
        print("\n[skip] images not downloaded. Add --images once the")
        print("       annotation structure has been confirmed below.")

    return inspect(ann_path)


if __name__ == "__main__":
    raise SystemExit(main())
