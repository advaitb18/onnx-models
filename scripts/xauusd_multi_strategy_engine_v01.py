#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = (
    ROOT / "config" / "multi_strategy_portfolio.json"
)


# ============================================================
# GENERIC HELPERS
# ============================================================

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def resolve(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def discover_mt5_common() -> Path:
    override = os.environ.get("XAU_MT5_COMMON_FILES")

    if override:
        p = Path(override)
        if p.exists():
            return p

    users = Path("/mnt/c/Users")

    if users.exists():
        candidates = []

        for user in users.iterdir():
            p = (
                user
                / "AppData"
                / "Roaming"
                / "MetaQuotes"
                / "Terminal"
                / "Common"
                / "Files"
            )

            if p.exists():
                candidates.append(p)

        # Prefer whichever installation already contains our exporter.
        for p in candidates:
            if (
                p
                / "xau_signals"
                / "xauusd_bar_export_heartbeat.txt"
            ).exists():
                return p

        if candidates:
            return candidates[0]

    raise RuntimeError(
        "Could not discover MT5 Common Files. "
        "Set XAU_MT5_COMMON_FILES."
    )


def run_command(command: list[str]) -> None:
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=900,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "Builder failed:\n"
            f"COMMAND={command}\n"
            f"STDOUT:\n{proc.stdout[-4000:]}\n"
            f"STDERR:\n{proc.stderr[-4000:]}"
        )


def feature_order(path: Path) -> list[str]:
    raw = load_json(path)

    if isinstance(raw, list):
        values = raw
    else:
        values = (
            raw.get("feature_order")
            or raw.get("features")
            or raw.get("columns")
            or []
        )

    return [str(x) for x in values]


def probability(outputs: list[Any]) -> float:
    for output in reversed(outputs):
        if (
            isinstance(output, list)
            and output
            and isinstance(output[0], dict)
        ):
            for key in (1, "1", True):
                if key in output[0]:
                    return float(output[0][key])

        arr = np.asarray(output)

        if (
            arr.ndim == 2
            and arr.shape[1] >= 2
            and arr.dtype.kind in "fiu"
        ):
            return float(arr[0, 1])

    raise RuntimeError(
        "Could not identify positive-class probability."
    )


def onnx_score(model_path: Path, x: np.ndarray) -> float:
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    inp = session.get_inputs()[0]

    dtype = (
        np.float64
        if "double" in inp.type
        else np.float32
    )

    score = probability(
        session.run(
            None,
            {inp.name: x.astype(dtype)},
        )
    )

    if not math.isfinite(score):
        raise RuntimeError("Non-finite model score.")

    return float(score)


def score_column(df: pd.DataFrame) -> str:
    candidates = []

    for col in df.columns:
        low = col.lower()

        if "prob" in low:
            candidates.append(col)
        elif low.endswith("_rf"):
            candidates.append(col)
        elif "prediction" in low:
            candidates.append(col)
        elif "score" in low and "rank" not in low:
            candidates.append(col)

    if not candidates:
        raise RuntimeError(
            "Could not determine distribution score column."
        )

    for col in candidates:
        if "rf" in col.lower():
            return col

    return candidates[0]


def rank_pct(path: Path, value: float) -> float:
    df = pd.read_parquet(path)

    values = pd.to_numeric(
        df[score_column(df)],
        errors="coerce",
    ).to_numpy(float)

    values = np.sort(
        values[np.isfinite(values)]
    )

    if len(values) < 100:
        raise RuntimeError(
            f"Distribution too small: {len(values)}"
        )

    return float(
        np.searchsorted(
            values,
            value,
            side="right",
        ) / len(values)
    )


def atr_from_row(row: pd.Series) -> float:
    for col in ("atr_14", "atr14", "atr"):
        if col in row.index:
            value = pd.to_numeric(
                row[col],
                errors="coerce",
            )

            if (
                pd.notna(value)
                and math.isfinite(float(value))
                and float(value) > 0
            ):
                return float(value)

    raise RuntimeError("Valid ATR not found.")


# ============================================================
# ADAPTER: ONNX_RANK
# ============================================================

def run_onnx_rank(
    strategy_id: str,
    strategy: dict[str, Any],
) -> dict[str, Any]:

    builder = strategy["builder"]
    model_cfg = strategy["model"]
    decision = strategy["decision"]

    run_command(builder["command"])

    data_path = resolve(
        builder["output_path"]
    )

    order = feature_order(
        resolve(model_cfg["feature_order_path"])
    )

    available = set(
        pq.read_schema(data_path).names
    )

    extras = [
        "timestamp",
        "datetime",
        "atr_14",
        "atr14",
        "atr",
        "confirmed_last_swing_high",
        "confirmed_last_swing_low",
    ]

    requested = list(dict.fromkeys(
        col
        for col in order + extras
        if col in available
    ))

    df = pd.read_parquet(
        data_path,
        columns=requested,
    )

    # Preserve V03H compatibility repair.
    from scripts.p8b0b_live_feature_builder_callable_smoke_test import (
        add_live_missing_v03h_features,
    )

    df = add_live_missing_v03h_features(df)

    row = df.iloc[-1]

    timestamp_col = next(
        (
            c
            for c in ("timestamp", "datetime")
            if c in row.index
        ),
        None,
    )

    if timestamp_col is None:
        raise RuntimeError(
            f"{strategy_id}: timestamp missing."
        )

    bar_id = pd.to_datetime(
        row[timestamp_col],
        utc=True,
    ).isoformat()

    values = pd.to_numeric(
        row[order],
        errors="coerce",
    ).to_numpy(np.float32)

    if not np.isfinite(values).all():
        bad = [
            order[i]
            for i in np.flatnonzero(
                ~np.isfinite(values)
            )[:20]
        ]

        raise RuntimeError(
            f"{strategy_id}: bad features: {bad}"
        )

    score = onnx_score(
        resolve(model_cfg["path"]),
        values.reshape(1, -1),
    )

    rank = rank_pct(
        resolve(
            model_cfg["score_distribution_path"]
        ),
        score,
    )

    threshold = float(
        decision["threshold"]
    )

    return {
        "strategy_id": strategy_id,
        "bar_id": bar_id,
        "score": score,
        "rank_pct": rank,
        "threshold": threshold,
        "qualified": rank >= threshold,
        "atr": atr_from_row(row),
    }


# ============================================================
# ADAPTER: EXTERNAL_SUMMARY
# ============================================================

def run_external_summary(
    strategy_id: str,
    strategy: dict[str, Any],
) -> dict[str, Any]:

    builder = strategy["builder"]
    decision = strategy["decision"]

    run_command(builder["command"])

    summary_path = resolve(
        builder["summary_path"]
    )

    summary = load_json(summary_path)

    if summary.get("status") != "PASS":
        raise RuntimeError(
            f"{strategy_id}: external pipeline "
            f"status={summary.get('status')}"
        )

    score_field = decision.get(
        "score_field",
        "prob_buy",
    )

    score = float(
        summary[score_field]
    )

    threshold = float(
        decision["threshold"]
    )

    qualified = score >= threshold

    candidate_field = decision.get(
        "candidate_field"
    )

    if candidate_field in summary:
        qualified = (
            qualified
            and bool(summary[candidate_field])
        )

    bar_id = pd.to_datetime(
        summary["latest_timestamp"],
        utc=True,
    ).isoformat()

    # V05H feature pipeline already creates M5 ATR.
    atr_candidates = [
        ROOT
        / "data"
        / "v05h2"
        / "work"
        / "features"
        / "timeframe=M5"
        / "xauusd_M5_features.parquet",

        ROOT
        / "data"
        / "v05h2"
        / "work"
        / "features"
        / "mtf_aligned"
        / "base_timeframe=M5"
        / "xauusd_M5_mtf_features.parquet",
    ]

    atr = None

    for path in atr_candidates:
        if not path.exists():
            continue

        available = set(
            pq.read_schema(path).names
        )

        cols = [
            c
            for c in (
                "timestamp",
                "atr_14",
                "atr14",
                "atr",
            )
            if c in available
        ]

        if "timestamp" not in cols:
            continue

        frame = pd.read_parquet(
            path,
            columns=cols,
        )

        if frame.empty:
            continue

        atr = atr_from_row(
            frame.iloc[-1]
        )
        break

    if atr is None:
        raise RuntimeError(
            f"{strategy_id}: unable to obtain ATR."
        )

    return {
        "strategy_id": strategy_id,
        "bar_id": bar_id,
        "score": score,
        "rank_pct": None,
        "threshold": threshold,
        "qualified": qualified,
        "atr": atr,
    }


# ============================================================
# SIGNAL WRITING
# ============================================================

def signal_text(fields: dict[str, Any]) -> str:
    return "\n".join(
        f"{key}={value}"
        for key, value in fields.items()
    ) + "\n"


def publish_signal(
    inbox: Path,
    strategy_id: str,
    strategy: dict[str, Any],
    result: dict[str, Any],
) -> Path:

    trade = strategy["trade"]

    now = int(time.time())

    raw_id = (
        f"{strategy_id}|"
        f"{result['bar_id']}|"
        f"{result['score']:.10f}"
    )

    signal_id = hashlib.sha256(
        raw_id.encode()
    ).hexdigest()[:24].upper()

    fields = {
        "schema_version": 2,
        "signal_id": signal_id,
        "strategy_id": strategy_id,
        "model_id": strategy["model_id"],
        "symbol": strategy["symbol"],
        "timeframe": strategy["timeframe"],
        "decision": strategy["direction"],
        "entry_type": trade["entry_type"],
        "created_at_epoch": now,
        "expires_at_epoch": (
            now
            + int(
                trade.get(
                    "signal_expiry_seconds",
                    240,
                )
            )
        ),
        "signal_bar_timestamp": result["bar_id"],
        "model_score": f"{result['score']:.10f}",
        "rank_pct": (
            ""
            if result["rank_pct"] is None
            else f"{result['rank_pct']:.10f}"
        ),
        "decision_threshold": (
            f"{result['threshold']:.10f}"
        ),
        "atr": f"{result['atr']:.10f}",
        "tp_atr": trade["tp_atr"],
        "sl_atr": trade["sl_atr"],
        "max_hold_bars": trade["max_hold_bars"],
        "magic": trade["magic"],
    }

    inbox.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_path = (
        inbox
        / f"{strategy_id}_{signal_id}.signal"
    )

    tmp = final_path.with_suffix(
        ".signal.tmp"
    )

    tmp.write_text(
        signal_text(fields),
        encoding="utf-8",
    )

    os.replace(
        tmp,
        final_path,
    )

    # --------------------------------------------------------
    # Rebuild deterministic queue index for MT5 EA.
    # One relative signal filename per line.
    # --------------------------------------------------------
    queue_index = inbox.parent / "queue.txt"

    signal_files = sorted(
        inbox.glob("*.signal"),
        key=lambda p: (p.stat().st_mtime_ns, p.name),
    )

    queue_text = "".join(
        f"{p.name}\n"
        for p in signal_files
    )

    queue_tmp = queue_index.with_suffix(".txt.tmp")

    queue_tmp.write_text(
        queue_text,
        encoding="utf-8",
    )

    os.replace(
        queue_tmp,
        queue_index,
    )

    return final_path



# ============================================================
# GENERIC MT5 STRATEGY CONTRACT REGISTRY
# ============================================================

def write_strategy_contracts(
    mt5_common: Path,
    cfg: dict[str, Any],
) -> Path:

    portfolio_root = (
        mt5_common
        / "xau_signals"
        / "portfolio"
    )

    portfolio_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        portfolio_root
        / "strategy_contracts.csv"
    )

    lines = [
        "magic,strategy_id,timeframe,max_hold_bars\n"
    ]

    seen_magics: set[int] = set()

    for strategy_id, strategy in cfg["strategies"].items():

        if not strategy.get("enabled", False):
            continue

        trade = strategy.get("trade", {})

        magic = int(
            trade.get("magic", 0)
        )

        timeframe = str(
            strategy.get("timeframe", "")
        ).upper()

        max_hold_bars = int(
            trade.get("max_hold_bars", 0)
        )

        if magic <= 0:
            raise RuntimeError(
                f"{strategy_id}: invalid magic={magic}"
            )

        if magic in seen_magics:
            raise RuntimeError(
                f"Duplicate magic number: {magic}"
            )

        if not timeframe:
            raise RuntimeError(
                f"{strategy_id}: timeframe missing"
            )

        if max_hold_bars <= 0:
            raise RuntimeError(
                f"{strategy_id}: invalid max_hold_bars="
                f"{max_hold_bars}"
            )

        seen_magics.add(magic)

        lines.append(
            f"{magic},"
            f"{strategy_id},"
            f"{timeframe},"
            f"{max_hold_bars}\n"
        )

    tmp = path.with_suffix(".csv.tmp")

    tmp.write_text(
        "".join(lines),
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )

    return path


# ============================================================
# ENGINE
# ============================================================

ADAPTERS = {
    "ONNX_RANK": run_onnx_rank,
    "EXTERNAL_SUMMARY": run_external_summary,
}

def rebuild_queue_index(inbox: Path) -> Path:
    inbox.mkdir(parents=True, exist_ok=True)

    queue_index = inbox.parent / "queue.txt"

    signal_files = sorted(
        inbox.glob("*.signal"),
        key=lambda p: (p.stat().st_mtime_ns, p.name),
    )

    queue_text = "".join(
        f"{p.name}\n"
        for p in signal_files
    )

    tmp = queue_index.with_suffix(".txt.tmp")

    tmp.write_text(
        queue_text,
        encoding="utf-8",
    )

    os.replace(
        tmp,
        queue_index,
    )

    return queue_index

def cycle(config_path: Path) -> dict[str, Any]:

    cfg = load_json(config_path)

    mt5_common = discover_mt5_common()

    portfolio = cfg["portfolio"]

    strategy_contract_path = write_strategy_contracts(
        mt5_common,
        cfg,
    )

    inbox = (
        mt5_common
        / portfolio["signal_inbox"]
    )
    rebuild_queue_index(inbox)
    state_path = resolve(
        portfolio["state_file"]
    )

    state = (
        load_json(state_path)
        if state_path.exists()
        else {"strategies": {}}
    )

    state.setdefault(
        "strategies",
        {},
    )

    cycle_results = []

    for strategy_id, strategy in (
        cfg["strategies"].items()
    ):

        if not strategy.get(
            "enabled",
            False,
        ):
            continue

        adapter_name = strategy.get(
            "adapter"
        )

        adapter = ADAPTERS.get(
            adapter_name
        )

        if adapter is None:
            cycle_results.append({
                "strategy_id": strategy_id,
                "status": "UNSUPPORTED_ADAPTER",
                "adapter": adapter_name,
            })
            continue

        try:
            result = adapter(
                strategy_id,
                strategy,
            )

            strategy_state = (
                state["strategies"]
                .setdefault(
                    strategy_id,
                    {},
                )
            )

            if (
                strategy_state.get(
                    "last_bar_id"
                )
                == result["bar_id"]
            ):
                cycle_results.append({
                    **result,
                    "status": "NO_NEW_CLOSED_BAR",
                })
                continue

            strategy_state.update({
                "last_bar_id": result["bar_id"],
                "last_score": result["score"],
                "last_rank_pct": result["rank_pct"],
                "updated_utc": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
            })

            atomic_json(
                state_path,
                state,
            )

            if not result["qualified"]:
                cycle_results.append({
                    **result,
                    "status": "HOLD",
                })
                continue

            signal_path = publish_signal(
                inbox,
                strategy_id,
                strategy,
                result,
            )

            cycle_results.append({
                **result,
                "status": "SIGNAL_PUBLISHED",
                "signal_path": str(signal_path),
            })

        except Exception as exc:
            cycle_results.append({
                "strategy_id": strategy_id,
                "status": "ERROR",
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            })
    rebuild_queue_index(inbox)
    return {
        "engine_version": (
            cfg.get(
                "engine_version",
                "XAUUSD_MULTI_STRATEGY_ENGINE_V01",
            )
        ),
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "mt5_common_files": str(
            mt5_common
        ),
        "independent_strategy_execution": True,
        "strategy_results": cycle_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
    )

    parser.add_argument(
        "--once",
        action="store_true",
    )

    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    config_path = Path(
        args.config
    ).resolve()

    if args.once:
        print(
            json.dumps(
                cycle(config_path),
                indent=2,
            )
        )
        return 0

    while True:
        try:
            print(
                json.dumps(
                    cycle(config_path),
                    indent=2,
                ),
                flush=True,
            )
        except Exception as exc:
            print(
                f"ENGINE_ERROR="
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

        time.sleep(
            max(5, args.poll_seconds)
        )


if __name__ == "__main__":
    raise SystemExit(main())
