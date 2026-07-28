"""
snapshot.py
-----------
Atomic JSON snapshot writer + latest-snapshot reader for the order-book
conviction layer.

Snapshots are written to data/orderbook/BTC_snapshot_{YYYYmmdd_HHMMSS}.json
following the pattern established by scripts/btc/astro_notifier.py (which
writes data/astro/astro_{date}.json), but with two improvements:

  1. Atomic writes via tempfile + os.replace — a process crash mid-write
     never corrupts a snapshot. (The astro_notifier writes directly.)
  2. load_latest_snapshot() helper for the main pipeline to read the most
     recent snapshot at inference time.

Public API:
    write_snapshot(snapshot_dict, snapshot_dir=None, config_path="config/orderbook.yaml")
        -> Path

    load_latest_snapshot(snapshot_dir=None, config_path="config/orderbook.yaml")
        -> Optional[Dict]

    cleanup_old_snapshots(snapshot_dir=None, config_path="config/orderbook.yaml")
        -> int   (number of files deleted)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_yaml(relative_path: str) -> Dict[str, Any]:
    root = Path(__file__).resolve().parent.parent.parent.parent
    full = root / relative_path
    if not full.exists():
        raise FileNotFoundError(f"Config not found: {full}")
    with open(full, "r") as f:
        return yaml.safe_load(f) or {}


def _resolve_snapshot_dir(
    snapshot_dir: Optional[str],
    config_path: str,
) -> Path:
    """Resolve snapshot directory from arg, then config, then default."""
    if snapshot_dir:
        p = Path(snapshot_dir)
    else:
        try:
            cfg = _load_yaml(config_path)
            snapshot_dir = cfg.get("storage", {}).get("snapshot_dir", "data/orderbook")
        except Exception:
            snapshot_dir = "data/orderbook"
        p = Path(snapshot_dir)
    # If relative, anchor at project root so behaviour is CWD-independent
    if not p.is_absolute():
        p = ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def _filename_for_now(now: Optional[datetime] = None) -> str:
    """Return e.g. BTC_snapshot_20260726_143005.json (UTC)."""
    now = now or datetime.now(timezone.utc)
    return f"BTC_snapshot_{now.strftime('%Y%m%d_%H%M%S')}.json"


def write_snapshot(
    snapshot_dict: Dict[str, Any],
    snapshot_dir: Optional[str] = None,
    config_path: str = "config/orderbook.yaml",
) -> Path:
    """
    Write a snapshot dict to disk atomically.

    Writes to a temp file in the same directory, then os.replace()s it
    onto the final path. This guarantees no half-written file is ever
    visible to readers even if the process is killed mid-write.
    """
    out_dir = _resolve_snapshot_dir(snapshot_dir, config_path)
    fname = _filename_for_now()
    final_path = out_dir / fname

    # Atomic write: temp file in same dir (so os.replace is atomic on POSIX)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_snapshot_", suffix=".json", dir=str(out_dir))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(snapshot_dict, f, indent=2, default=str)
        os.replace(tmp_path, final_path)
    except Exception:
        # Clean up the temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return final_path


def load_latest_snapshot(
    snapshot_dir: Optional[str] = None,
    config_path: str = "config/orderbook.yaml",
    max_age_seconds: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load the most recent snapshot file from the snapshot directory.

    Args:
        max_age_seconds: if set, ignore snapshots older than this many
            seconds. Useful so the pipeline doesn't act on stale data
            when the poll job has died.

    Returns None if no snapshots exist or all are stale.
    """
    out_dir = _resolve_snapshot_dir(snapshot_dir, config_path)
    candidates = sorted(out_dir.glob("BTC_snapshot_*.json"), reverse=True)
    if not candidates:
        return None

    now = datetime.now(timezone.utc)
    for path in candidates:
        # Quick freshness check via mtime
        if max_age_seconds is not None:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            age = (now - mtime).total_seconds()
            if age > max_age_seconds:
                continue
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            continue
    return None


def cleanup_old_snapshots(
    snapshot_dir: Optional[str] = None,
    config_path: str = "config/orderbook.yaml",
) -> int:
    """
    Delete snapshot files older than storage.cleanup_days.

    Returns the number of files deleted. If cleanup_days is 0 or absent,
    no files are deleted.
    """
    try:
        cfg = _load_yaml(config_path)
    except Exception:
        return 0
    cleanup_days = int(cfg.get("storage", {}).get("cleanup_days", 0))
    if cleanup_days <= 0:
        return 0

    out_dir = _resolve_snapshot_dir(snapshot_dir, config_path)
    cutoff = datetime.now(timezone.utc) - timedelta(days=cleanup_days)
    deleted = 0
    for path in out_dir.glob("BTC_snapshot_*.json"):
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if mtime < cutoff:
                path.unlink()
                deleted += 1
        except Exception:
            continue
    return deleted


if __name__ == "__main__":
    # Smoke test: write a fake snapshot, then load it back
    test_data = {
        "test": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "exchanges": {"binance": {"spot_price": 60000.0}},
    }
    p = write_snapshot(test_data)
    print(f"Wrote: {p}")
    latest = load_latest_snapshot()
    print(f"Loaded: {latest is not None}")
    if latest:
        print(f"  test flag: {latest.get('test')}")
