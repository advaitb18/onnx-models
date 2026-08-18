from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def discover_mt5_common() -> Path:
    override = os.environ.get(
        "XAU_MT5_COMMON_FILES"
    )

    if override:
        path = Path(override).expanduser()

        if path.exists():
            return path.resolve()

        raise RuntimeError(
            "XAU_MT5_COMMON_FILES points to a "
            f"missing directory: {path}"
        )

    users_root = Path("/mnt/c/Users")

    if users_root.exists():
        candidates = []

        for user in users_root.iterdir():
            candidate = (
                user
                / "AppData"
                / "Roaming"
                / "MetaQuotes"
                / "Terminal"
                / "Common"
                / "Files"
            )

            if candidate.exists():
                candidates.append(candidate)

        # Prefer the MT5 installation already running
        # the XAUUSD bar exporter.
        for candidate in candidates:
            heartbeat = (
                candidate
                / "xau_signals"
                / "xauusd_bar_export_heartbeat.txt"
            )

            if heartbeat.exists():
                return candidate.resolve()

        if len(candidates) == 1:
            return candidates[0].resolve()

        if candidates:
            raise RuntimeError(
                "Multiple MT5 Common Files directories "
                "were detected. Set XAU_MT5_COMMON_FILES "
                "to choose the correct terminal."
            )

    raise RuntimeError(
        "MT5 Common Files directory was not found. "
        "Set XAU_MT5_COMMON_FILES."
    )
