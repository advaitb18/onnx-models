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

REPORT_DIR = PROJECT_ROOT / "reports" / "data_quality"
LOG_DIR = PROJECT_ROOT / "logs" / "python"

REPORT_MD = REPORT_DIR / "p3f_advanced_indicator_pack_report.md"
SUMMARY_CSV = REPORT_DIR / "p3f_advanced_indicator_pack_summary.csv"
LOG_JSONL = LOG_DIR / "p3f_advanced_indicator_pack.jsonl"

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


def load_registry() -> dict[str, Any]:
    if not FEATURE_REGISTRY_YAML.exists():
        raise FileNotFoundError(f"Missing feature registry: {FEATURE_REGISTRY_YAML}")
    with FEATURE_REGISTRY_YAML.open("r", encoding="utf-8") as f:
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


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def add_macd(out: pd.DataFrame) -> None:
    ema_12 = ema(out["close"], 12)
    ema_26 = ema(out["close"], 26)
    out["macd_line"] = ema_12 - ema_26
    out["macd_signal"] = ema(out["macd_line"], 9)
    out["macd_hist"] = out["macd_line"] - out["macd_signal"]
    out["macd_hist_slope"] = out["macd_hist"].diff(3)
    out["macd_bullish"] = (out["macd_line"] > out["macd_signal"]).astype("int8")


def add_bollinger(out: pd.DataFrame) -> None:
    mid = out["close"].rolling(20, min_periods=20).mean()
    std = out["close"].rolling(20, min_periods=20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std

    out["bb_mid_20"] = mid
    out["bb_upper_20"] = upper
    out["bb_lower_20"] = lower
    out["bb_width_20"] = (upper - lower) / mid.replace(0, np.nan)
    out["bb_percent_b_20"] = (out["close"] - lower) / (upper - lower).replace(0, np.nan)
    out["bb_squeeze_20"] = (
        out["bb_width_20"]
        < out["bb_width_20"].rolling(100, min_periods=100).quantile(0.20)
    ).astype("int8")


def add_adx(out: pd.DataFrame) -> None:
    high = out["high"]
    low = out["low"]
    close = out["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=out.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=out.index,
    )

    if "tr" in out.columns:
        tr = out["tr"]
    else:
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    out["plus_di_14"] = plus_di
    out["minus_di_14"] = minus_di
    out["adx_14"] = adx
    out["di_bullish"] = (out["plus_di_14"] > out["minus_di_14"]).astype("int8")
    out["adx_strong_trend"] = (out["adx_14"] >= 25).astype("int8")


def add_stochastic_williams(out: pd.DataFrame) -> None:
    low_14 = out["low"].rolling(14, min_periods=14).min()
    high_14 = out["high"].rolling(14, min_periods=14).max()
    denom = (high_14 - low_14).replace(0, np.nan)

    k = 100 * (out["close"] - low_14) / denom
    d = k.rolling(3, min_periods=3).mean()

    out["stoch_k_14"] = k
    out["stoch_d_3"] = d
    out["stoch_overbought"] = (out["stoch_k_14"] >= 80).astype("int8")
    out["stoch_oversold"] = (out["stoch_k_14"] <= 20).astype("int8")

    out["williams_r_14"] = -100 * (high_14 - out["close"]) / denom


def add_cci(out: pd.DataFrame) -> None:
    tp = (out["high"] + out["low"] + out["close"]) / 3
    sma_tp = tp.rolling(20, min_periods=20).mean()

    mean_dev = (tp - sma_tp).abs().rolling(20, min_periods=20).mean()
    out["cci_20"] = (tp - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))


def add_channels(out: pd.DataFrame) -> None:
    high_20 = out["high"].rolling(20, min_periods=20).max()
    low_20 = out["low"].rolling(20, min_periods=20).min()

    out["donchian_high_20"] = high_20
    out["donchian_low_20"] = low_20
    out["donchian_mid_20"] = (high_20 + low_20) / 2
    out["donchian_width_20"] = (high_20 - low_20) / out["close"].replace(0, np.nan)
    out["donchian_breakout_up_20"] = (out["close"] > high_20.shift(1)).astype("int8")
    out["donchian_breakout_down_20"] = (out["close"] < low_20.shift(1)).astype("int8")

    ema_20 = out["ema_20"] if "ema_20" in out.columns else ema(out["close"], 20)
    atr_14 = out["atr_14"] if "atr_14" in out.columns else out["tr"].ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    out["keltner_mid_20"] = ema_20
    out["keltner_upper_20"] = ema_20 + 2 * atr_14
    out["keltner_lower_20"] = ema_20 - 2 * atr_14
    out["keltner_width_20"] = (out["keltner_upper_20"] - out["keltner_lower_20"]) / ema_20.replace(0, np.nan)


def add_regime_and_distance(out: pd.DataFrame) -> None:
    out["close_to_ema20_pct"] = (out["close"] - out["ema_20"]) / out["ema_20"].replace(0, np.nan)
    out["close_to_ema50_pct"] = (out["close"] - out["ema_50"]) / out["ema_50"].replace(0, np.nan)
    out["close_to_ema200_pct"] = (out["close"] - out["ema_200"]) / out["ema_200"].replace(0, np.nan)

    out["atr_pct_of_close"] = out["atr_14"] / out["close"].replace(0, np.nan)
    out["atr_percentile_200"] = out["atr_pct_of_close"].rolling(200, min_periods=200).rank(pct=True)

    out["volatility_regime_low"] = (out["atr_percentile_200"] <= 0.33).astype("int8")
    out["volatility_regime_mid"] = (
        (out["atr_percentile_200"] > 0.33) & (out["atr_percentile_200"] < 0.66)
    ).astype("int8")
    out["volatility_regime_high"] = (out["atr_percentile_200"] >= 0.66).astype("int8")

    out["trend_regime_bull"] = (
        (out["close"] > out["ema_50"])
        & (out["ema_50"] > out["ema_200"])
        & (out["ema_50_slope"] > 0)
    ).astype("int8")

    out["trend_regime_bear"] = (
        (out["close"] < out["ema_50"])
        & (out["ema_50"] < out["ema_200"])
        & (out["ema_50_slope"] < 0)
    ).astype("int8")

    out["trend_regime_sideways"] = (
        (out["trend_regime_bull"] == 0) & (out["trend_regime_bear"] == 0)
    ).astype("int8")


def add_vwap_proxy(out: pd.DataFrame) -> None:
    typical = (out["high"] + out["low"] + out["close"]) / 3
    pv = typical * out["volume"]

    vol_20 = out["volume"].rolling(20, min_periods=20).sum()
    vol_50 = out["volume"].rolling(50, min_periods=50).sum()

    out["rolling_vwap_20"] = pv.rolling(20, min_periods=20).sum() / vol_20.replace(0, np.nan)
    out["rolling_vwap_50"] = pv.rolling(50, min_periods=50).sum() / vol_50.replace(0, np.nan)
    out["close_to_vwap20_pct"] = (out["close"] - out["rolling_vwap_20"]) / out["rolling_vwap_20"].replace(0, np.nan)
    out["close_to_vwap50_pct"] = (out["close"] - out["rolling_vwap_50"]) / out["rolling_vwap_50"].replace(0, np.nan)


def add_advanced_indicators(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    before_cols = set(out.columns)

    add_macd(out)
    add_bollinger(out)
    add_adx(out)
    add_stochastic_williams(out)
    add_cci(out)
    add_channels(out)
    add_regime_and_distance(out)
    add_vwap_proxy(out)

    # Replace inf with NaN so later safety/audit logic can handle cleanly.
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Dtype optimization.
    int_like = [
        "macd_bullish",
        "bb_squeeze_20",
        "di_bullish",
        "adx_strong_trend",
        "stoch_overbought",
        "stoch_oversold",
        "donchian_breakout_up_20",
        "donchian_breakout_down_20",
        "volatility_regime_low",
        "volatility_regime_mid",
        "volatility_regime_high",
        "trend_regime_bull",
        "trend_regime_bear",
        "trend_regime_sideways",
    ]

    for col in int_like:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype("int8")

    float_cols = out.select_dtypes(include=["float64"]).columns
    out[float_cols] = out[float_cols].astype("float32")

    added_cols = [c for c in out.columns if c not in before_cols]
    return out, added_cols


def summarize(timeframe: str, before: pd.DataFrame, after: pd.DataFrame, added_cols: list[str], path: Path) -> dict[str, Any]:
    null_cells_added = int(after[added_cols].isna().sum().sum()) if added_cols else 0
    inf_cells = int(np.isinf(after.select_dtypes(include=[np.number]).to_numpy()).sum())

    safe_rows = int(after["is_feature_row_safe"].sum()) if "is_feature_row_safe" in after.columns else None

    summary = {
        "timeframe": timeframe,
        "rows": int(len(after)),
        "columns_before": int(len(before.columns)),
        "columns_after": int(len(after.columns)),
        "added_column_count": int(len(added_cols)),
        "added_columns": added_cols,
        "null_cells_added_columns": null_cells_added,
        "infinite_numeric_cells": inf_cells,
        "safe_rows_existing_mask": safe_rows,
        "output_file": rel(path),
        "status": "OK" if len(added_cols) > 0 and inf_cells == 0 else "FAILED",
        "created_at_utc": now_utc(),
    }

    return summary


def update_registry(registry: dict[str, Any], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    registry.setdefault("feature_groups", {})

    registry["feature_groups"]["advanced_indicators_p3f"] = [
        "macd_line",
        "macd_signal",
        "macd_hist",
        "macd_hist_slope",
        "macd_bullish",
        "bb_mid_20",
        "bb_upper_20",
        "bb_lower_20",
        "bb_width_20",
        "bb_percent_b_20",
        "bb_squeeze_20",
        "plus_di_14",
        "minus_di_14",
        "adx_14",
        "di_bullish",
        "adx_strong_trend",
        "stoch_k_14",
        "stoch_d_3",
        "stoch_overbought",
        "stoch_oversold",
        "williams_r_14",
        "cci_20",
        "donchian_high_20",
        "donchian_low_20",
        "donchian_mid_20",
        "donchian_width_20",
        "donchian_breakout_up_20",
        "donchian_breakout_down_20",
        "keltner_mid_20",
        "keltner_upper_20",
        "keltner_lower_20",
        "keltner_width_20",
        "close_to_ema20_pct",
        "close_to_ema50_pct",
        "close_to_ema200_pct",
        "atr_pct_of_close",
        "atr_percentile_200",
        "volatility_regime_low",
        "volatility_regime_mid",
        "volatility_regime_high",
        "trend_regime_bull",
        "trend_regime_bear",
        "trend_regime_sideways",
        "rolling_vwap_20",
        "rolling_vwap_50",
        "close_to_vwap20_pct",
        "close_to_vwap50_pct",
    ]

    registry.setdefault("feature_policy", {})
    registry["feature_policy"]["advanced_indicator_pack_added"] = True
    registry["feature_policy"]["advanced_indicator_pack_phase"] = "P3F"
    registry["feature_policy"]["advanced_indicator_pack_uses_pandas_numpy_only"] = True
    registry["feature_policy"]["updated_at_utc"] = now_utc()

    registry.setdefault("timeframes", {})
    for s in summaries:
        tf = s["timeframe"]
        registry["timeframes"].setdefault(tf, {})
        registry["timeframes"][tf]["advanced_indicator_pack_p3f"] = {
            "status": s["status"],
            "added_column_count": s["added_column_count"],
            "added_columns": s["added_columns"],
            "updated_at_utc": s["created_at_utc"],
        }
        registry["timeframes"][tf]["feature_count"] = s["columns_after"] - 8

    return registry


def write_report(summaries: list[dict[str, Any]]) -> None:
    lines = []
    lines.append("# P3F Advanced Indicator Pack Report")
    lines.append("")
    lines.append(f"Created UTC: `{now_utc()}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Added advanced technical indicator features to existing feature parquet files.")
    lines.append("- Used custom pandas/numpy implementations.")
    lines.append("- Did not create signals, labels, ML models, or backtests.")
    lines.append("- Existing `is_feature_row_safe` mask was preserved.")
    lines.append("")
    lines.append("## Added indicator families")
    lines.append("")
    lines.append("- MACD")
    lines.append("- Bollinger Bands")
    lines.append("- ADX / DI+ / DI-")
    lines.append("- Stochastic")
    lines.append("- Williams %R")
    lines.append("- CCI")
    lines.append("- Donchian Channels")
    lines.append("- Keltner Channels")
    lines.append("- Rolling VWAP proxy")
    lines.append("- ATR percentile and volatility regime")
    lines.append("- Trend regime")
    lines.append("- Distance-from-EMA features")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| TF | Status | Rows | Cols Before | Cols After | Added | Inf Cells | Added Null Cells |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

    for s in summaries:
        lines.append(
            f"| {s['timeframe']} | {s['status']} | {s['rows']} | {s['columns_before']} | "
            f"{s['columns_after']} | {s['added_column_count']} | {s['infinite_numeric_cells']} | "
            f"{s['null_cells_added_columns']} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Initial rolling indicator windows naturally create nulls.")
    lines.append("- Next phase should rerun feature quality audit and optionally update safety rules for new indicators.")
    lines.append("- These indicators are features only; they are not a strategy by themselves.")

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    registry = load_registry()
    summaries: list[dict[str, Any]] = []

    table = Table(title="P3F Advanced Indicator Pack")
    table.add_column("TF")
    table.add_column("Rows")
    table.add_column("Before")
    table.add_column("After")
    table.add_column("Added")
    table.add_column("Inf")
    table.add_column("Status")

    for tf in TIMEFRAMES:
        console.print(f"[bold]Adding advanced indicators for {tf}[/bold]")

        before, path = load_features(tf)
        after, added_cols = add_advanced_indicators(before)

        after.to_parquet(path, index=False)

        summary = summarize(tf, before, after, added_cols, path)
        summaries.append(summary)
        log_event(summary)

        table.add_row(
            tf,
            str(summary["rows"]),
            str(summary["columns_before"]),
            str(summary["columns_after"]),
            str(summary["added_column_count"]),
            str(summary["infinite_numeric_cells"]),
            summary["status"],
        )

    registry = update_registry(registry, summaries)
    write_registry(registry)

    summary_df = pd.DataFrame(summaries)
    summary_df["added_columns"] = summary_df["added_columns"].apply(lambda x: json.dumps(x, default=str))
    summary_df.to_csv(SUMMARY_CSV, index=False)

    write_report(summaries)

    console.print(table)
    console.print("[bold green]P3F advanced indicator pack complete.[/bold green]")
    console.print(f"Report: {REPORT_MD}")
    console.print(f"Summary: {SUMMARY_CSV}")
    console.print(f"Feature registry: {FEATURE_REGISTRY_YAML}")


if __name__ == "__main__":
    main()
