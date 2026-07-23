"""
Scheduler config loader.
Reads config/scheduler.yaml — single source of truth for all timed jobs.
Systemd remains the executor; this module provides programmatic access
to job definitions for tooling and future stock pipeline integration.
"""

from __future__ import annotations

import os
from typing import Dict, Any, Optional

import yaml


def _default_config_path() -> str:
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))
    return os.path.join(base, "config", "scheduler.yaml")


def load_jobs(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load all job definitions from scheduler.yaml."""
    path = config_path or _default_config_path()
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("jobs", {})


def get_job(name: str, config_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get a single job definition by name."""
    return load_jobs(config_path).get(name)
