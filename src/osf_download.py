"""Resumable downloader for OSF projects.

Written because osfclient's `clone` has two properties that make it a poor fit
for a multi-gigabyte download on a home connection:

  * it offers no resume, so a dropped connection means starting over
  * it raises on the first network error, discarding everything

This module downloads each file to a `.part` file in a persistent directory
and resumes with an HTTP Range request if interrupted. Nothing is ever deleted
on failure, so re-running the script picks up where it stopped.

Only `requests` is needed, which osfclient already depends on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import requests

OSF_API = "https://api.osf.io/v2"

# requests.iter_content accumulates up to chunk_size before yielding, so any
# data still buffered when a connection drops is lost. Measured retention of
# the bytes actually sent before a drop:
#
#     1 MiB chunk  ->   0.0%      64 KiB chunk -> 93.6%
#   256 KiB chunk  ->  74.9%       8 KiB chunk -> 99.5%
#
# 64 KiB caps the loss per interruption at 64 KB, which is nothing against a
# multi-gigabyte download, while keeping the iteration count low.
CHUNK = 1 << 16  # 64 KiB

DEFAULT_RETRIES = 6
DEFAULT_TIMEOUT = 60


@dataclass
class OSFFile:
    name: str
    size: int
    download_url: str


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def list_files(node: str, timeout: int = DEFAULT_TIMEOUT) -> list[OSFFile]:
    """List files in a public OSF project's osfstorage, following pagination."""
    url = f"{OSF_API}/nodes/{node}/files/osfstorage/"
    files: list[OSFFile] = []

    while url:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()

        for item in payload.get("data", []):
            attrs = item.get("attributes", {})
            if attrs.get("kind") != "file":
                continue
            link = item.get("links", {}).get("download")
            if not link:
                continue
            files.append(
                OSFFile(
                    name=attrs.get("name", "unnamed"),
                    size=int(attrs.get("size") or 0),
                    download_url=link,
                )
            )

        url = payload.get("links", {}).get("next")

    return files


def _open_stream(
    url: str, offset: int, timeout: int
) -> tuple[requests.Response, bool]:
    """Open a download stream, requesting a byte range when resuming.

    Returns the response and whether the server honoured the range request.
    """
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    resp = requests.get(url, headers=headers, stream=True, timeout=timeout)
    resp.raise_for_status()
    # 206 means the range was honoured; 200 means the server is sending the
    # whole file and any existing partial data must be discarded.
    return resp, resp.status_code == 206


def download_file(
    f: OSFFile,
    dest_dir: Path,
    retries: int = DEFAULT_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    progress: Callable[[str], None] = print,
) -> Path:
    """Download one file, resuming if a partial download exists.

    Raises the last exception if every attempt fails.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / f.name
    part = dest_dir / (f.name + ".part")

    if final.is_file() and (f.size == 0 or final.stat().st_size == f.size):
        progress(f"  [have] {f.name} ({human(final.stat().st_size)})")
        return final

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        offset = part.stat().st_size if part.is_file() else 0

        if f.size and offset > f.size:
            # Corrupt leftover; start clean.
            part.unlink()
            offset = 0

        if f.size and offset == f.size:
            part.replace(final)
            progress(f"  [done] {f.name}")
            return final

        try:
            resp, ranged = _open_stream(f.download_url, offset, timeout)

            if offset and not ranged:
                progress("  [warn] server ignored resume, restarting file")
                offset = 0

            mode = "ab" if offset else "wb"
            downloaded = offset
            started = time.monotonic()
            last_report = 0.0

            with resp, part.open(mode) as fh:
                for chunk in resp.iter_content(chunk_size=CHUNK):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    downloaded += len(chunk)

                    now = time.monotonic()
                    if now - last_report >= 2.0:
                        # Push to disk so an abrupt drop cannot lose buffered
                        # writes, keeping the resume point accurate.
                        fh.flush()
                        last_report = now
                        elapsed = max(now - started, 1e-6)
                        rate = (downloaded - offset) / elapsed
                        if f.size:
                            pct = 100.0 * downloaded / f.size
                            progress(
                                f"    {f.name}: {human(downloaded)}/{human(f.size)}"
                                f" ({pct:.1f}%) at {human(rate)}/s",
                            )
                        else:
                            progress(
                                f"    {f.name}: {human(downloaded)} at {human(rate)}/s"
                            )

            if f.size and part.stat().st_size != f.size:
                raise IOError(
                    f"size mismatch: got {part.stat().st_size}, expected {f.size}"
                )

            part.replace(final)
            progress(f"  [done] {f.name} ({human(final.stat().st_size)})")
            return final

        except (requests.RequestException, IOError) as exc:
            last_error = exc
            have = part.stat().st_size if part.is_file() else 0
            if attempt == retries:
                break
            backoff = min(2 ** attempt, 60)
            progress(
                f"  [retry {attempt}/{retries}] {type(exc).__name__}: {exc}\n"
                f"           kept {human(have)}, resuming in {backoff}s"
            )
            time.sleep(backoff)

    raise RuntimeError(
        f"Failed to download {f.name} after {retries} attempts. "
        f"Partial data is preserved at {part}; re-run to resume."
    ) from last_error


def download_project(
    node: str,
    dest_dir: Path,
    retries: int = DEFAULT_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    progress: Callable[[str], None] = print,
) -> Iterator[Path]:
    """Download every file in an OSF project, yielding completed paths."""
    progress(f"[osf] listing files in project {node}")
    files = list_files(node, timeout=timeout)

    if not files:
        raise RuntimeError(f"No files found in OSF project {node}")

    total = sum(f.size for f in files)
    progress(f"[osf] {len(files)} files, {human(total)} total")
    for f in files:
        progress(f"       {f.name:16} {human(f.size)}")

    for f in files:
        yield download_file(f, dest_dir, retries, timeout, progress)
