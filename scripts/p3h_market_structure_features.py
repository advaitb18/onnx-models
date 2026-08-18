from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = PROJECT_ROOT / "data" / "features" / "xauusd"
FEATURE_REGISTRY_YAML = FEATURE_ROOT / "feature_registry.yaml"
FEATURE_REGISTRY_JSON = FEATURE_ROOT / "feature_registry.json"
SAFETY_CONTRACT_YAML = FEATURE_ROOT / "feature_safety_contract.yaml"

REPORT_DIR = PROJECT_ROOT / "reports" / "data_quality"
LOG_DIR = PROJECT_ROOT / "logs" / "python"

REPORT_MD = REPORT_DIR / "p3h_market_structure_features_report.md"
SUMMARY_CSV = REPORT_DIR / "p3h_market_structure_features_summary.csv"
EVENT_COUNTS_CSV = REPORT_DIR / "p3h_market_structure_event_counts.csv"
LOG_JSONL = LOG_DIR / "p3h_market_structure_features.jsonl"

TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]

# Swing windows are intentionally timeframe-aware.
# Larger timeframes need smaller windows because they already aggregate structure.
SWING_WINDOWS = {
    "M1": 5,
    "M5": 5,
    "M15": 4,
    "M30": 4,
    "H1": 3,
    "H4": 3,
    "D1": 2,
}

STRUCTURE_LOOKBACK = {
    "M1": 150,
    "M5": 120,
    "M15": 100,
    "M30": 80,
    "H1": 60,
    "H4": 40,
    "D1": 30,
}

console = Console()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def log_event(record: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_registry(registry: dict[str, Any]) -> None:
    FEATURE_REGISTRY_YAML.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    FEATURE_REGISTRY_JSON.write_text(json.dumps(registry, indent=2, default=str), encoding="utf-8")


def load_features(timeframe: str) -> tuple[pd.DataFrame, Path]:
    path = FEATURE_ROOT / f"timeframe={timeframe}" / f"xauusd_{timeframe}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")

    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return df, path


def detect_swings(df: pd.DataFrame, window: int) -> tuple[pd.Series, pd.Series]:
    high = df["high"]
    low = df["low"]

    # Centered swing confirmation. This means the last `window` bars cannot know confirmed swings yet.
    rolling_high = high.rolling(window=(2 * window + 1), center=True, min_periods=(2 * window + 1)).max()
    rolling_low = low.rolling(window=(2 * window + 1), center=True, min_periods=(2 * window + 1)).min()

    swing_high = (high == rolling_high) & rolling_high.notna()
    swing_low = (low == rolling_low) & rolling_low.notna()

    # Avoid marking both on the same candle unless it is truly an extreme wide candle.
    both = swing_high & swing_low
    if both.any():
        candle_range = df["high"] - df["low"]
        range_threshold = candle_range.rolling(100, min_periods=20).quantile(0.95)
        allow_both = candle_range >= range_threshold
        swing_high = swing_high & (~both | allow_both)
        swing_low = swing_low & (~both | allow_both)

    return swing_high.fillna(False), swing_low.fillna(False)


def forward_fill_last_swing(df: pd.DataFrame, swing_high: pd.Series, swing_low: pd.Series) -> pd.DataFrame:
    out = df.copy()

    out["swing_high_price"] = np.where(swing_high, out["high"], np.nan)
    out["swing_low_price"] = np.where(swing_low, out["low"], np.nan)

    out["last_swing_high"] = pd.Series(out["swing_high_price"]).ffill()
    out["last_swing_low"] = pd.Series(out["swing_low_price"]).ffill()

    out["prev_swing_high"] = pd.Series(out["swing_high_price"]).ffill().shift(1)
    out["prev_swing_low"] = pd.Series(out["swing_low_price"]).ffill().shift(1)

    out["bars_since_swing_high"] = compute_bars_since_event(pd.Series(swing_high, index=out.index))
    out["bars_since_swing_low"] = compute_bars_since_event(pd.Series(swing_low, index=out.index))

    return out


def compute_bars_since_event(event: pd.Series) -> pd.Series:
    values = event.fillna(False).astype(bool).to_numpy()
    result = np.empty(len(values), dtype=np.int32)

    last_idx = -1_000_000_000
    for i, v in enumerate(values):
        if v:
            last_idx = i
            result[i] = 0
        else:
            result[i] = i - last_idx

    result[result > 100_000_000] = 1_000_000_000
    return pd.Series(result, index=event.index)


def add_structure_features(df: pd.DataFrame, timeframe: str) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    before_cols = set(out.columns)

    window = SWING_WINDOWS[timeframe]
    lookback = STRUCTURE_LOOKBACK[timeframe]

    swing_high, swing_low = detect_swings(out, window)

    out["structure_swing_window"] = window
    out["structure_lookback_bars"] = lookback

    out["swing_high"] = swing_high.astype("int8")
    out["swing_low"] = swing_low.astype("int8")

    out = forward_fill_last_swing(out, swing_high, swing_low)

    # Higher-high / lower-high only on swing-high bars.
    out["higher_high"] = (
        (out["swing_high"] == 1)
        & out["swing_high_price"].notna()
        & out["prev_swing_high"].notna()
        & (out["swing_high_price"] > out["prev_swing_high"])
    ).astype("int8")

    out["lower_high"] = (
        (out["swing_high"] == 1)
        & out["swing_high_price"].notna()
        & out["prev_swing_high"].notna()
        & (out["swing_high_price"] < out["prev_swing_high"])
    ).astype("int8")

    # Higher-low / lower-low only on swing-low bars.
    out["higher_low"] = (
        (out["swing_low"] == 1)
        & out["swing_low_price"].notna()
        & out["prev_swing_low"].notna()
        & (out["swing_low_price"] > out["prev_swing_low"])
    ).astype("int8")

    out["lower_low"] = (
        (out["swing_low"] == 1)
        & out["swing_low_price"].notna()
        & out["prev_swing_low"].notna()
        & (out["swing_low_price"] < out["prev_swing_low"])
    ).astype("int8")

    # Break of structure: close breaks prior confirmed swing level.
    prior_swing_high = out["last_swing_high"].shift(1)
    prior_swing_low = out["last_swing_low"].shift(1)

    out["break_of_structure_bull"] = (
        prior_swing_high.notna()
        & (out["close"] > prior_swing_high)
        & (out["close"].shift(1) <= prior_swing_high)
    ).astype("int8")

    out["break_of_structure_bear"] = (
        prior_swing_low.notna()
        & (out["close"] < prior_swing_low)
        & (out["close"].shift(1) >= prior_swing_low)
    ).astype("int8")

    # Liquidity sweep candidate:
    # wick takes swing level but candle closes back inside.
    out["liquidity_sweep_high"] = (
        prior_swing_high.notna()
        & (out["high"] > prior_swing_high)
        & (out["close"] < prior_swing_high)
    ).astype("int8")

    out["liquidity_sweep_low"] = (
        prior_swing_low.notna()
        & (out["low"] < prior_swing_low)
        & (out["close"] > prior_swing_low)
    ).astype("int8")

    # Structure state from rolling events.
    bull_events = (
        out["break_of_structure_bull"].rolling(lookback, min_periods=1).sum()
        + out["higher_high"].rolling(lookback, min_periods=1).sum()
        + out["higher_low"].rolling(lookback, min_periods=1).sum()
    )

    bear_events = (
        out["break_of_structure_bear"].rolling(lookback, min_periods=1).sum()
        + out["lower_high"].rolling(lookback, min_periods=1).sum()
        + out["lower_low"].rolling(lookback, min_periods=1).sum()
    )

    out["structure_bull_score"] = bull_events.astype("float32")
    out["structure_bear_score"] = bear_events.astype("float32")
    out["structure_score_net"] = (out["structure_bull_score"] - out["structure_bear_score"]).astype("float32")

    out["structure_trend"] = np.select(
        [
            out["structure_score_net"] >= 2,
            out["structure_score_net"] <= -2,
        ],
        [
            "BULL",
            "BEAR",
        ],
        default="RANGE",
    )

    out["structure_trend_bull"] = (out["structure_trend"] == "BULL").astype("int8")
    out["structure_trend_bear"] = (out["structure_trend"] == "BEAR").astype("int8")
    out["structure_trend_range"] = (out["structure_trend"] == "RANGE").astype("int8")

    # CHoCH candidate:
    # first opposite break after recent structure regime.
    prev_trend = pd.Series(out["structure_trend"]).shift(1)
    out["choch_bull"] = (
        (out["break_of_structure_bull"] == 1)
        & prev_trend.eq("BEAR")
    ).astype("int8")

    out["choch_bear"] = (
        (out["break_of_structure_bear"] == 1)
        & prev_trend.eq("BULL")
    ).astype("int8")

    # Distances normalized by ATR and close.
    out["distance_to_last_swing_high"] = out["last_swing_high"] - out["close"]
    out["distance_to_last_swing_low"] = out["close"] - out["last_swing_low"]

    out["distance_to_last_swing_high_pct"] = out["distance_to_last_swing_high"] / out["close"].replace(0, np.nan)
    out["distance_to_last_swing_low_pct"] = out["distance_to_last_swing_low"] / out["close"].replace(0, np.nan)

    atr = out["atr_14"].replace(0, np.nan) if "atr_14" in out.columns else pd.Series(np.nan, index=out.index)

    out["distance_to_last_swing_high_atr"] = out["distance_to_last_swing_high"] / atr
    out["distance_to_last_swing_low_atr"] = out["distance_to_last_swing_low"] / atr

    out["near_last_swing_high"] = (
        out["distance_to_last_swing_high_atr"].abs().le(0.5)
        & out["distance_to_last_swing_high_atr"].notna()
    ).astype("int8")

    out["near_last_swing_low"] = (
        out["distance_to_last_swing_low_atr"].abs().le(0.5)
        & out["distance_to_last_swing_low_atr"].notna()
    ).astype("int8")

    # Avoid infinities.
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Dtype cleanup.
    int8_cols = [
        "swing_high",
        "swing_low",
        "higher_high",
        "lower_high",
        "higher_low",
        "lower_low",
        "break_of_structure_bull",
        "break_of_structure_bear",
        "liquidity_sweep_high",
        "liquidity_sweep_low",
        "structure_trend_bull",
        "structure_trend_bear",
        "structure_trend_range",
        "choch_bull",
        "choch_bear",
        "near_last_swing_high",
        "near_last_swing_low",
    ]

    for col in int8_cols:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype("int8")

    int32_cols = [
        "structure_swing_window",
        "structure_lookback_bars",
        "bars_since_swing_high",
        "bars_since_swing_low",
    ]

    for col in int32_cols:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype("int32")

    float_cols = out.select_dtypes(include=["float64"]).columns
    out[float_cols] = out[float_cols].astype("float32")

    added_cols = [c for c in out.columns if c not in before_cols]
    return out, added_cols


def summarize(timeframe: str, before: pd.DataFrame, after: pd.DataFrame, added_cols: list[str], path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    safe_rows = int(after["is_feature_row_safe"].fillna(0).astype(int).sum()) if "is_feature_row_safe" in after.columns else None

    event_cols = [
        "swing_high",
        "swing_low",
        "higher_high",
        "higher_low",
        "lower_high",
        "lower_low",
        "break_of_structure_bull",
        "break_of_structure_bear",
        "choch_bull",
        "choch_bear",
        "liquidity_sweep_high",
        "liquidity_sweep_low",
        "structure_trend_bull",
        "structure_trend_bear",
        "structure_trend_range",
        "near_last_swing_high",
        "near_last_swing_low",
    ]

    event_rows = []
    for col in event_cols:
        if col in after.columns:
            count = int(after[col].fillna(0).astype(int).sum())
            event_rows.append({
                "timeframe": timeframe,
                "event": col,
                "rows": count,
                "row_ratio": round(count / max(1, len(after)), 8),
            })

    event_df = pd.DataFrame(event_rows)

    null_cells_added = int(after[added_cols].isna().sum().sum()) if added_cols else 0
    inf_cells = int(np.isinf(after.select_dtypes(include=[np.number]).to_numpy()).sum())

    summary = {
        "timeframe": timeframe,
        "rows": int(len(after)),
        "columns_before": int(len(before.columns)),
        "columns_after": int(len(after.columns)),
        "added_column_count": int(len(added_cols)),
        "added_columns": added_cols,
        "safe_rows": safe_rows,
        "safe_row_ratio": round(safe_rows / max(1, len(after)), 8) if safe_rows is not None else None,
        "swing_window": SWING_WINDOWS[timeframe],
        "structure_lookback": STRUCTURE_LOOKBACK[timeframe],
        "swing_high_count": int(after["swing_high"].sum()),
        "swing_low_count": int(after["swing_low"].sum()),
        "bos_bull_count": int(after["break_of_structure_bull"].sum()),
        "bos_bear_count": int(after["break_of_structure_bear"].sum()),
        "choch_bull_count": int(after["choch_bull"].sum()),
        "choch_bear_count": int(after["choch_bear"].sum()),
        "liquidity_sweep_high_count": int(after["liquidity_sweep_high"].sum()),
        "liquidity_sweep_low_count": int(after["liquidity_sweep_low"].sum()),
        "null_cells_added_columns": null_cells_added,
        "infinite_numeric_cells": inf_cells,
        "output_file": rel(path),
        "status": "OK" if len(added_cols) > 0 and inf_cells == 0 else "FAILED",
        "created_at_utc": now_utc(),
    }

    return summary, event_df


def update_registry(registry: dict[str, Any], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    registry.setdefault("feature_groups", {})
    registry["feature_groups"]["market_structure_p3h"] = [
        "structure_swing_window",
        "structure_lookback_bars",
        "swing_high",
        "swing_low",
        "swing_high_price",
        "swing_low_price",
        "last_swing_high",
        "last_swing_low",
        "prev_swing_high",
        "prev_swing_low",
        "bars_since_swing_high",
        "bars_since_swing_low",
        "higher_high",
        "lower_high",
        "higher_low",
        "lower_low",
        "break_of_structure_bull",
        "break_of_structure_bear",
        "choch_bull",
        "choch_bear",
        "liquidity_sweep_high",
        "liquidity_sweep_low",
        "structure_bull_score",
        "structure_bear_score",
        "structure_score_net",
        "structure_trend",
        "structure_trend_bull",
        "structure_trend_bear",
        "structure_trend_range",
        "distance_to_last_swing_high",
        "distance_to_last_swing_low",
        "distance_to_last_swing_high_pct",
        "distance_to_last_swing_low_pct",
        "distance_to_last_swing_high_atr",
        "distance_to_last_swing_low_atr",
        "near_last_swing_high",
        "near_last_swing_low",
    ]

    registry.setdefault("feature_policy", {})
    registry["feature_policy"]["market_structure_p3h_added"] = True
    registry["feature_policy"]["market_structure_uses_confirmed_centered_swings"] = True
    registry["feature_policy"]["market_structure_note"] = (
        "Centered swing features require future bars for confirmation. "
        "They are acceptable for historical research/backtesting if shifted/confirmed correctly in label logic. "
        "Live implementation must use delayed confirmed swings only."
    )
    registry["feature_policy"]["downstream_required_filter"] = "is_feature_row_safe == 1"
    registry["feature_policy"]["updated_at_utc"] = now_utc()

    registry.setdefault("timeframes", {})
    for s in summaries:
        tf = s["timeframe"]
        registry["timeframes"].setdefault(tf, {})
        registry["timeframes"][tf]["market_structure_p3h"] = {
            "status": s["status"],
            "swing_window": s["swing_window"],
            "structure_lookback": s["structure_lookback"],
            "swing_high_count": s["swing_high_count"],
            "swing_low_count": s["swing_low_count"],
            "bos_bull_count": s["bos_bull_count"],
            "bos_bear_count": s["bos_bear_count"],
            "choch_bull_count": s["choch_bull_count"],
            "choch_bear_count": s["choch_bear_count"],
            "liquidity_sweep_high_count": s["liquidity_sweep_high_count"],
            "liquidity_sweep_low_count": s["liquidity_sweep_low_count"],
            "updated_at_utc": s["created_at_utc"],
        }
        registry["timeframes"][tf]["feature_count"] = s["columns_after"] - 8

    return registry


def write_report(summaries: list[dict[str, Any]], event_df: pd.DataFrame) -> None:
    lines = []
    lines.append("# P3H Market Structure Features Report")
    lines.append("")
    lines.append(f"Created UTC: `{now_utc()}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Added market-structure features to existing XAUUSD feature parquet files.")
    lines.append("- Added confirmed swing highs/lows, BOS, CHoCH candidates, structure trend, and liquidity sweep candidates.")
    lines.append("- Preserved existing safety contract and `is_feature_row_safe` filter.")
    lines.append("- No labels, signals, models, or backtests were created.")
    lines.append("")
    lines.append("## Important live-trading note")
    lines.append("")
    lines.append("This phase uses centered confirmed swings for research. Centered swings require future bars to confirm a swing.")
    lines.append("For live trading, the signal engine must use delayed confirmed swings only. This will be enforced in the live signal phase.")
    lines.append("")
    lines.append("## Downstream safety rule")
    lines.append("")
    lines.append("```python")
    lines.append("df = df[df['is_feature_row_safe'] == 1]")
    lines.append("```")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| TF | Status | Rows | Cols Before | Cols After | Added | Swing Window | BOS Bull | BOS Bear | CHoCH Bull | CHoCH Bear | Inf Cells |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for s in summaries:
        lines.append(
            f"| {s['timeframe']} | {s['status']} | {s['rows']} | {s['columns_before']} | "
            f"{s['columns_after']} | {s['added_column_count']} | {s['swing_window']} | "
            f"{s['bos_bull_count']} | {s['bos_bear_count']} | {s['choch_bull_count']} | "
            f"{s['choch_bear_count']} | {s['infinite_numeric_cells']} |"
        )

    lines.append("")
    lines.append("## Event counts")
    lines.append("")
    for tf in [s["timeframe"] for s in summaries]:
        lines.append(f"### {tf}")
        lines.append("")
        sub = event_df[event_df["timeframe"] == tf]
        for _, row in sub.iterrows():
            lines.append(f"- `{row['event']}`: rows=`{int(row['rows'])}`, ratio=`{round(float(row['row_ratio']), 8)}`")
        lines.append("")

    lines.append("## Output files")
    lines.append("")
    lines.append(f"- Summary CSV: `{rel(SUMMARY_CSV)}`")
    lines.append(f"- Event counts CSV: `{rel(EVENT_COUNTS_CSV)}`")
    lines.append(f"- Feature registry: `{rel(FEATURE_REGISTRY_YAML)}`")

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    registry = load_yaml(FEATURE_REGISTRY_YAML)
    safety_contract = load_yaml(SAFETY_CONTRACT_YAML)

    required_filter = safety_contract.get("downstream_required_filter")
    if required_filter != "is_feature_row_safe == 1":
        raise ValueError("Safety contract does not contain required downstream filter.")

    summaries = []
    event_frames = []

    table = Table(title="P3H Market Structure Features")
    table.add_column("TF")
    table.add_column("Rows")
    table.add_column("Before")
    table.add_column("After")
    table.add_column("Added")
    table.add_column("BOS Bull")
    table.add_column("BOS Bear")
    table.add_column("Status")

    for tf in TIMEFRAMES:
        console.print(f"[bold]Building market structure features for {tf}[/bold]")

        before, path = load_features(tf)
        after, added_cols = add_structure_features(before, tf)

        after.to_parquet(path, index=False)

        summary, event_df = summarize(tf, before, after, added_cols, path)
        summaries.append(summary)
        event_frames.append(event_df)

        log_event(summary)

        table.add_row(
            tf,
            str(summary["rows"]),
            str(summary["columns_before"]),
            str(summary["columns_after"]),
            str(summary["added_column_count"]),
            str(summary["bos_bull_count"]),
            str(summary["bos_bear_count"]),
            summary["status"],
        )

    all_events = pd.concat(event_frames, ignore_index=True)

    summary_df = pd.DataFrame(summaries)
    summary_df["added_columns"] = summary_df["added_columns"].apply(lambda x: json.dumps(x, default=str))
    summary_df.to_csv(SUMMARY_CSV, index=False)
    all_events.to_csv(EVENT_COUNTS_CSV, index=False)

    registry = update_registry(registry, summaries)
    write_registry(registry)

    write_report(summaries, all_events)

    console.print(table)
    console.print("[bold green]P3H market structure features complete.[/bold green]")
    console.print(f"Report: {REPORT_MD}")
    console.print(f"Summary: {SUMMARY_CSV}")
    console.print(f"Events: {EVENT_COUNTS_CSV}")
    console.print(f"Feature registry: {FEATURE_REGISTRY_YAML}")


if __name__ == "__main__":
    main()
