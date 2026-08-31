"""Download the Chess Pieces dataset from Roboflow Universe.

Usage
-----
    export ROBOFLOW_API_KEY="your_key_here"
    python src/download_data.py

A free Roboflow "Public" account is sufficient. No credit card is required,
and a free key can download public Universe datasets.

If you would rather not use the API at all, download the YOLOv8 export
manually from the dataset page, unzip it into data/raw/chess-pieces/, and
skip this script. inspect_data.py works either way.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import config


def download_via_api(
    api_key: str,
    dest: Path,
    workspace: str = config.ROBOFLOW_WORKSPACE,
    project: str = config.ROBOFLOW_PROJECT,
    version: int = config.ROBOFLOW_VERSION,
    fmt: str = config.ROBOFLOW_FORMAT,
    overwrite: bool = False,
) -> Path:
    """Fetch the dataset and return the directory it was written to."""
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit(
            "The roboflow package is not installed.\n"
            "Run: pip install roboflow"
        )

    if dest.exists():
        if not overwrite:
            print(f"[skip] {dest} already exists. Use --overwrite to replace it.")
            return dest
        print(f"[clean] removing existing {dest}")
        shutil.rmtree(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"[roboflow] {workspace}/{project} v{version} -> {fmt}")
    rf = Roboflow(api_key=api_key)
    ds = (
        rf.workspace(workspace)
        .project(project)
        .version(version)
        .download(fmt, location=str(dest))
    )

    location = Path(getattr(ds, "location", dest))
    print(f"[done] dataset at {location}")
    return location


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=config.RAW_DIR / "chess-pieces",
        help="Where to write the dataset (default: data/raw/chess-pieces)",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=config.ROBOFLOW_VERSION,
        help="Roboflow dataset version number",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and re-download if the destination already exists",
    )
    args = parser.parse_args()

    config.ensure_dirs()

    api_key = config.get_roboflow_key()
    if not api_key:
        sys.exit(
            f"No API key found.\n\n"
            f"Set it with:\n"
            f'    export {config.ROBOFLOW_API_KEY_ENV}="your_key_here"\n\n'
            f"or create a .env file at {config.REPO_ROOT / '.env'} containing:\n"
            f"    {config.ROBOFLOW_API_KEY_ENV}=your_key_here\n\n"
            f"Get a free key: sign up at roboflow.com, then find it under\n"
            f"Settings -> API Keys in your workspace."
        )

    location = download_via_api(
        api_key=api_key,
        dest=args.dest,
        version=args.version,
        overwrite=args.overwrite,
    )

    print("\nNext step:")
    print(f"    python src/inspect_data.py --data {location}")


if __name__ == "__main__":
    main()
