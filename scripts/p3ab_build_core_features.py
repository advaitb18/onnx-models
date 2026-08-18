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
REGISTRY_PATH = PROJECT_ROOT / "data" / "clean" / "xauusd" / "canonical_data_registry.yaml"
FEATURE_ROOT = PROJECT_ROOT / "data" / "features" / "xauusd"
REPORT_DIR = PROJECT_ROOT / "reports" / "data_quality"
LOG_DIR = PROJECT_ROOT / "logs" / "python"

FEATURE_REGISTRY_YAML = FEATURE_ROOT / "feature_registry.yaml"
FEATURE_REGISTRY_JSON = FEATURE_ROOT / "feature_registry.json"
REPORT_MD = REPORT_DIR / "p3ab_core_feature_report.md"
SUMMARY_CSV = REPORT_DIR / "p3ab_core_feature_summary.csv"
LOG_JSONL = LOG_DIR / "p3ab_core_features.jsonl"

TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]

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
        raise FileNotFoundError(f"Missing registry: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_ohlcv_from_registry(timeframe: str, registry: dict[str, Any]) -> pd.DataFrame:
    tf_info = registry.get("timeframes", {}).get(timeframe)
    if not tf_info:
        raise KeyError(f"Timeframe not found in registry: {timeframe}")

    if not tf_info.get("approved_for_research", False):
        raise ValueError(f"Timeframe not approved for research: {timeframe}")

    path = PROJECT_ROOT / tf_info["path"]
    if not path.exists():
        raise FileNotFoundError(f"Clean parquet missing: {path}")

    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    keep = ["timestamp", "symbol", "timeframe", "open", "high", "low", "close", "volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"{timeframe} missing required columns: {missing}")

    return df[keep].copy()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def add_core_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Returns
    out["ret_1"] = out["close"].pct_change(1)
    out["ret_3"] = out["close"].pct_change(3)
    out["ret_5"] = out["close"].pct_change(5)
    out["ret_10"] = out["close"].pct_change(10)
    out["log_ret_1"] = np.log(out["close"] / out["close"].shift(1))

    # Candle/range structure
    out["candle_range"] = out["high"] - out["low"]
    out["candle_body"] = (out["close"] - out["open"]).abs()
    out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]
    out["body_to_range"] = out["candle_body"] / out["candle_range"].replace(0, np.nan)
    out["close_position_in_range"] = (out["close"] - out["low"]) / out["candle_range"].replace(0, np.nan)
    out["is_bullish"] = (out["close"] > out["open"]).astype("int8")
    out["is_bearish"] = (out["close"] < out["open"]).astype("int8")
    out["is_zero_range"] = (out["candle_range"] == 0).astype("int8")

    # Moving averages
    out["sma_20"] = out["close"].rolling(20, min_periods=20).mean()
    out["sma_50"] = out["close"].rolling(50, min_periods=50).mean()
    out["ema_20"] = ema(out["close"], 20)
    out["ema_50"] = ema(out["close"], 50)
    out["ema_200"] = ema(out["close"], 200)

    out["close_above_ema_20"] = (out["close"] > out["ema_20"]).astype("int8")
    out["close_above_ema_50"] = (out["close"] > out["ema_50"]).astype("int8")
    out["close_above_ema_200"] = (out["close"] > out["ema_200"]).astype("int8")

    # Trend
    out["ema_20_slope"] = out["ema_20"].diff(5)
    out["ema_50_slope"] = out["ema_50"].diff(5)
    out["ema_20_above_ema_50"] = (out["ema_20"] > out["ema_50"]).astype("int8")
    out["ema_50_above_ema_200"] = (out["ema_50"] > out["ema_200"]).astype("int8")

    # Volatility
    out["tr"] = true_range(out)
    out["atr_14"] = out["tr"].ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    out["range_pct"] = out["candle_range"] / out["close"].replace(0, np.nan)
    out["rolling_vol_20"] = out["log_ret_1"].rolling(20, min_periods=20).std()
    out["rolling_vol_50"] = out["log_ret_1"].rolling(50, min_periods=50).std()

    # Momentum
    out["rsi_14"] = rsi(out["close"], 14)
    out["roc_5"] = out["close"].pct_change(5)
    out["roc_10"] = out["close"].pct_change(10)
    out["momentum_10"] = out["close"] - out["close"].shift(10)

    # Volume
    out["volume_sma_20"] = out["volume"].rolling(20, min_periods=20).mean()
    out["volume_ratio_20"] = out["volume"] / out["volume_sma_20"].replace(0, np.nan)

    # Time/session features in UTC
    out["hour"] = out["timestamp"].dt.hour.astype("int16")
    out["day_of_week"] = out["timestamp"].dt.dayofweek.astype("int16")
    out["month"] = out["timestamp"].dt.month.astype("int16")
    out["year"] = out["timestamp"].dt.year.astype("int16")

    # UTC session approximations. We can refine after broker-time comparison.
    out["is_asia_session"] = ((out["hour"] >= 0) & (out["hour"] < 7)).astype("int8")
    out["is_london_session"] = ((out["hour"] >= 7) & (out["hour"] < 16)).astype("int8")
    out["is_newyork_session"] = ((out["hour"] >= 12) & (out["hour"] < 21)).astype("int8")
    out["is_london_ny_overlap"] = ((out["hour"] >= 12) & (out["hour"] < 16)).astype("int8")

    return out


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    int_cols = [
        "is_bullish",
        "is_bearish",
        "is_zero_range",
        "close_above_ema_20",
        "close_above_ema_50",
        "close_above_ema_200",
        "ema_20_above_ema_50",
        "ema_50_above_ema_200",
        "is_asia_session",
        "is_london_session",
        "is_newyork_session",
        "is_london_ny_overlap",
    ]

    for col in int_cols:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype("int8")

    for col in ["hour", "day_of_week", "month", "year"]:
        if col in out.columns:
            out[col] = out[col].astype("int16")

    numeric_cols = out.select_dtypes(include=["float64"]).columns
    out[numeric_cols] = out[numeric_cols].astype("float32")

    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    base = {"timestamp", "symbol", "timeframe", "open", "high", "low", "close", "volume"}
    return [c for c in df.columns if c not in base]


def summarize_features(timeframe: str, raw: pd.DataFrame, features: pd.DataFrame, out_path: Path) -> dict[str, Any]:
    fcols = feature_columns(features)

    null_counts = features[fcols].isna().sum().sort_values(ascending=False)
    high_null = {
        col: int(val)
        for col, val in null_counts.items()
        if int(val) > 0
    }

    inf_count = int(np.isinf(features.select_dtypes(include=[np.number]).to_numpy()).sum())

    summary = {
        "timeframe": timeframe,
        "input_rows": int(len(raw)),
        "output_rows": int(len(features)),
        "feature_count": int(len(fcols)),
        "output_file": rel(out_path),
        "first_timestamp": str(features["timestamp"].min()) if not features.empty else None,
        "last_timestamp": str(features["timestamp"].max()) if not features.empty else None,
        "null_feature_cells": int(features[fcols].isna().sum().sum()) if fcols else 0,
        "infinite_numeric_cells": inf_count,
        "high_null_feature_counts": high_null,
        "status": "OK" if len(features) > 0 and len(fcols) > 0 else "FAILED",
        "created_at_utc": now_utc(),
    }
    return summary


def build_feature_registry(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    feature_groups = {
        "returns": ["ret_1", "ret_3", "ret_5", "ret_10", "log_ret_1"],
        "candle_range": [
            "candle_range", "candle_body", "upper_wick", "lower_wick",
            "body_to_range", "close_position_in_range",
            "is_bullish", "is_bearish", "is_zero_range",
        ],
        "moving_averages": [
            "sma_20", "sma_50", "ema_20", "ema_50", "ema_200",
            "close_above_ema_20", "close_above_ema_50", "close_above_ema_200",
        ],
        "trend": [
            "ema_20_slope", "ema_50_slope",
            "ema_20_above_ema_50", "ema_50_above_ema_200",
        ],
        "volatility": ["tr", "atr_14", "range_pct", "rolling_vol_20", "rolling_vol_50"],
        "momentum": ["rsi_14", "roc_5", "roc_10", "momentum_10"],
        "volume": ["volume_sma_20", "volume_ratio_20"],
        "time_session": [
            "hour", "day_of_week", "month", "year",
            "is_asia_session", "is_london_session", "is_newyork_session", "is_london_ny_overlap",
        ],
    }

    return {
        "registry_version": 1,
        "created_at_utc": now_utc(),
        "symbol": "XAUUSD",
        "source_data_registry": rel(REGISTRY_PATH),
        "feature_root": rel(FEATURE_ROOT),
        "feature_policy": {
            "phase": "P3A_P3B_CORE_FEATURES",
            "uses_only_custom_pandas_numpy_features": True,
            "does_not_create_signals": True,
            "does_not_create_labels": True,
            "does_not_train_models": True,
            "future_libraries": ["pandas-ta", "vectorbt"],
        },
        "feature_groups": feature_groups,
        "timeframes": {
            s["timeframe"]: {
                "path": s["output_file"],
                "rows": s["output_rows"],
                "feature_count": s["feature_count"],
                "status": s["status"],
                "first_timestamp": s["first_timestamp"],
                "last_timestamp": s["last_timestamp"],
            }
            for s in summaries
        },
    }


def write_report(summaries: list[dict[str, Any]], registry: dict[str, Any]) -> None:
    lines = []
    lines.append("# P3A/P3B Core Feature Report")
    lines.append("")
    lines.append(f"Created UTC: `{now_utc()}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Built first-pass custom pandas/numpy core features.")
    lines.append("- No pandas-ta/vectorbt yet.")
    lines.append("- No signals, labels, ML, or backtesting created in this phase.")
    lines.append("")
    lines.append("## Feature groups")
    lines.append("")
    for group, cols in registry["feature_groups"].items():
        lines.append(f"### {group}")
        lines.append("")
        for c in cols:
            lines.append(f"- `{c}`")
        lines.append("")
    lines.append("## Timeframe outputs")
    lines.append("")
    lines.append("| TF | Status | Rows | Features | Null Feature Cells | Infinite Cells | Output |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for s in summaries:
        lines.append(
            f"| {s['timeframe']} | {s['status']} | {s['output_rows']} | {s['feature_count']} | "
            f"{s['null_feature_cells']} | {s['infinite_numeric_cells']} | `{s['output_file']}` |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Initial rolling-window features naturally contain nulls at the beginning of each timeframe.")
    lines.append("- Later label/backtest phases should drop or mask warmup rows as needed.")
    lines.append("- Session features are UTC approximations and must be aligned with MT5 broker time later.")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    registry = load_yaml(REGISTRY_PATH)

    summaries: list[dict[str, Any]] = []

    table = Table(title="P3A/P3B Core Feature Build")
    table.add_column("TF")
    table.add_column("Rows")
    table.add_column("Features")
    table.add_column("Output")

    for tf in TIMEFRAMES:
        console.print(f"[bold]Building features for {tf}[/bold]")

        raw = load_ohlcv_from_registry(tf, registry)
        feat = add_core_features(raw)
        feat = optimize_dtypes(feat)

        out_dir = FEATURE_ROOT / f"timeframe={tf}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"xauusd_{tf}_features.parquet"

        feat.to_parquet(out_path, index=False)

        summary = summarize_features(tf, raw, feat, out_path)
        summaries.append(summary)
        log_event(summary)

        table.add_row(tf, str(summary["output_rows"]), str(summary["feature_count"]), summary["output_file"])

    feature_registry = build_feature_registry(summaries)

    FEATURE_REGISTRY_YAML.write_text(yaml.safe_dump(feature_registry, sort_keys=False), encoding="utf-8")
    FEATURE_REGISTRY_JSON.write_text(json.dumps(feature_registry, indent=2, default=str), encoding="utf-8")

    summary_df = pd.DataFrame(summaries)
    summary_df["high_null_feature_counts"] = summary_df["high_null_feature_counts"].apply(lambda x: json.dumps(x, default=str))
    summary_df.to_csv(SUMMARY_CSV, index=False)

    write_report(summaries, feature_registry)

    console.print(table)
    console.print("[bold green]P3A/P3B core features complete.[/bold green]")
    console.print(f"Feature registry: {FEATURE_REGISTRY_YAML}")
    console.print(f"Report: {REPORT_MD}")


if __name__ == "__main__":
    main()
