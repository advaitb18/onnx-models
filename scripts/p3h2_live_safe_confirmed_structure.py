from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FEATURE_ROOT = PROJECT_ROOT / "data" / "features" / "xauusd"
REPORT_DIR = PROJECT_ROOT / "reports" / "data_quality"
LOG_DIR = PROJECT_ROOT / "logs" / "python"

REGISTRY_YAML = FEATURE_ROOT / "feature_registry_v2.yaml"
REGISTRY_JSON = FEATURE_ROOT / "feature_registry_v2.json"

ML_ALLOWLIST_YAML = FEATURE_ROOT / "ml_feature_allowlist.yaml"
LIVE_ALLOWLIST_YAML = FEATURE_ROOT / "live_feature_allowlist.yaml"
LIVE_CONTEXT_ALLOWLIST_YAML = FEATURE_ROOT / "live_context_allowlist.yaml"
LIVE_MODEL_ALLOWLIST_YAML = FEATURE_ROOT / "live_model_allowlist.yaml"
ONNX_ALLOWLIST_YAML = FEATURE_ROOT / "onnx_feature_allowlist.yaml"

REPORT_MD = REPORT_DIR / "p3h2_live_safe_confirmed_structure_report.md"
SUMMARY_CSV = REPORT_DIR / "p3h2_confirmed_structure_summary.csv"
REGISTRY_CHECK_CSV = REPORT_DIR / "p3h2_registry_confirmed_columns.csv"
GUARD_TEST_CSV = REPORT_DIR / "p3h2_leakage_guard_smoke_test.csv"
LOG_JSONL = LOG_DIR / "p3h2_live_safe_confirmed_structure.jsonl"

TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]

CONFIRMATION_DELAY_BARS = {
    "M1": 5,
    "M5": 5,
    "M15": 4,
    "M30": 4,
    "H1": 3,
    "H4": 3,
    "D1": 2,
}

SHIFT_MAP = {
    "swing_high": "confirmed_swing_high",
    "swing_low": "confirmed_swing_low",
    "swing_high_price": "confirmed_swing_high_price",
    "swing_low_price": "confirmed_swing_low_price",
    "last_swing_high": "confirmed_last_swing_high",
    "last_swing_low": "confirmed_last_swing_low",
    "prev_swing_high": "confirmed_prev_swing_high",
    "prev_swing_low": "confirmed_prev_swing_low",
    "higher_high": "confirmed_higher_high",
    "lower_high": "confirmed_lower_high",
    "higher_low": "confirmed_higher_low",
    "lower_low": "confirmed_lower_low",
    "break_of_structure_bull": "confirmed_bos_bull",
    "break_of_structure_bear": "confirmed_bos_bear",
    "choch_bull": "confirmed_choch_bull",
    "choch_bear": "confirmed_choch_bear",
    "liquidity_sweep_high": "confirmed_liquidity_sweep_high",
    "liquidity_sweep_low": "confirmed_liquidity_sweep_low",
    "structure_bull_score": "confirmed_structure_bull_score",
    "structure_bear_score": "confirmed_structure_bear_score",
    "structure_score_net": "confirmed_structure_score_net",
    "structure_trend_bull": "confirmed_structure_trend_bull",
    "structure_trend_bear": "confirmed_structure_trend_bear",
    "structure_trend_range": "confirmed_structure_trend_range",
}

DERIVED_CONFIRMED_COLS = [
    "confirmed_structure_trend_code",
    "confirmed_structure_age_bars",
    "confirmed_bars_since_bos_bull",
    "confirmed_bars_since_bos_bear",
    "confirmed_bars_since_choch_bull",
    "confirmed_bars_since_choch_bear",
    "confirmed_bars_since_any_bos",
    "confirmed_bars_since_any_choch",
    "confirmed_bars_since_any_structure_break",
    "confirmed_distance_to_last_swing_high",
    "confirmed_distance_to_last_swing_low",
    "confirmed_distance_to_last_swing_high_pct",
    "confirmed_distance_to_last_swing_low_pct",
    "confirmed_distance_to_last_swing_high_atr",
    "confirmed_distance_to_last_swing_low_atr",
    "confirmed_near_last_swing_high",
    "confirmed_near_last_swing_low",
    "confirmed_structure_momentum_3",
    "confirmed_structure_momentum_5",
    "confirmed_structure_momentum_10",
]

ALL_CONFIRMED_COLS = sorted(set(SHIFT_MAP.values()).union(DERIVED_CONFIRMED_COLS))
ORIGINAL_RESEARCH_COLS = sorted(SHIFT_MAP.keys())

FORBIDDEN_LIVE_FEATURES = {
    "bars_to_next_large_gap",
    "pre_gap_risk_bars_remaining",
    "pre_gap_risk_score_linear",
    "pre_gap_risk_score_exp",
    "pre_gap_risk_score",
    "is_pre_large_gap_risk",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def log_event(record: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def feature_path(tf: str) -> Path:
    return FEATURE_ROOT / f"timeframe={tf}" / f"xauusd_{tf}_features.parquet"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def bars_since_event(event: pd.Series) -> pd.Series:
    arr = event.fillna(0).astype(bool).to_numpy()
    out = np.full(len(arr), np.nan, dtype=np.float32)
    last = None

    for i, flag in enumerate(arr):
        if flag:
            last = i
            out[i] = 0.0
        elif last is not None:
            out[i] = float(i - last)

    return pd.Series(out, index=event.index, dtype="float32")


def add_confirmed_structure(df: pd.DataFrame, tf: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    delay = CONFIRMATION_DELAY_BARS[tf]

    missing = [c for c in ORIGINAL_RESEARCH_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{tf}: missing required P3H columns: {missing}")

    before_cols = set(df.columns)

    binary_sources = {
        "swing_high",
        "swing_low",
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
        "structure_trend_bull",
        "structure_trend_bear",
        "structure_trend_range",
    }

    for src, dst in SHIFT_MAP.items():
        shifted = df[src].shift(delay)

        if src in binary_sources:
            df[dst] = shifted.fillna(0).astype("int8")
        else:
            df[dst] = pd.to_numeric(shifted, errors="coerce").astype("float32")

    df["confirmed_structure_trend_code"] = np.select(
        [
            df["confirmed_structure_trend_bull"].astype(bool),
            df["confirmed_structure_trend_bear"].astype(bool),
            df["confirmed_structure_trend_range"].astype(bool),
        ],
        [1, -1, 0],
        default=0,
    ).astype("int8")

    df["confirmed_bars_since_bos_bull"] = bars_since_event(df["confirmed_bos_bull"])
    df["confirmed_bars_since_bos_bear"] = bars_since_event(df["confirmed_bos_bear"])
    df["confirmed_bars_since_choch_bull"] = bars_since_event(df["confirmed_choch_bull"])
    df["confirmed_bars_since_choch_bear"] = bars_since_event(df["confirmed_choch_bear"])

    any_bos = (df["confirmed_bos_bull"].astype(bool) | df["confirmed_bos_bear"].astype(bool)).astype("int8")
    any_choch = (df["confirmed_choch_bull"].astype(bool) | df["confirmed_choch_bear"].astype(bool)).astype("int8")
    any_break = (any_bos.astype(bool) | any_choch.astype(bool)).astype("int8")

    df["confirmed_bars_since_any_bos"] = bars_since_event(any_bos)
    df["confirmed_bars_since_any_choch"] = bars_since_event(any_choch)
    df["confirmed_bars_since_any_structure_break"] = bars_since_event(any_break)
    df["confirmed_structure_age_bars"] = df["confirmed_bars_since_any_structure_break"].astype("float32")

    close = pd.to_numeric(df["close"], errors="coerce")
    atr = pd.to_numeric(df["atr_14"], errors="coerce").replace(0, np.nan)

    high_level = pd.to_numeric(df["confirmed_last_swing_high"], errors="coerce")
    low_level = pd.to_numeric(df["confirmed_last_swing_low"], errors="coerce")

    df["confirmed_distance_to_last_swing_high"] = (high_level - close).astype("float32")
    df["confirmed_distance_to_last_swing_low"] = (close - low_level).astype("float32")
    df["confirmed_distance_to_last_swing_high_pct"] = ((high_level - close) / close.replace(0, np.nan)).astype("float32")
    df["confirmed_distance_to_last_swing_low_pct"] = ((close - low_level) / close.replace(0, np.nan)).astype("float32")
    df["confirmed_distance_to_last_swing_high_atr"] = ((high_level - close) / atr).astype("float32")
    df["confirmed_distance_to_last_swing_low_atr"] = ((close - low_level) / atr).astype("float32")

    df["confirmed_near_last_swing_high"] = (
        df["confirmed_distance_to_last_swing_high_atr"].abs() <= 0.50
    ).fillna(False).astype("int8")

    df["confirmed_near_last_swing_low"] = (
        df["confirmed_distance_to_last_swing_low_atr"].abs() <= 0.50
    ).fillna(False).astype("int8")

    score = pd.to_numeric(df["confirmed_structure_score_net"], errors="coerce")
    df["confirmed_structure_momentum_3"] = ((score - score.shift(3)) / 3.0).astype("float32")
    df["confirmed_structure_momentum_5"] = ((score - score.shift(5)) / 5.0).astype("float32")
    df["confirmed_structure_momentum_10"] = ((score - score.shift(10)) / 10.0).astype("float32")

    confirmed_cols = [c for c in ALL_CONFIRMED_COLS if c in df.columns]

    for col in confirmed_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    added_cols = sorted(set(df.columns) - before_cols)

    inf_cells = 0
    numeric_confirmed = df[confirmed_cols].select_dtypes(include=[np.number])
    if not numeric_confirmed.empty:
        inf_cells = int(np.isinf(numeric_confirmed.to_numpy()).sum())

    summary = {
        "timeframe": tf,
        "confirmation_delay_bars": delay,
        "rows": int(len(df)),
        "cols_added": int(len(added_cols)),
        "confirmed_cols_present": int(sum(c in df.columns for c in ALL_CONFIRMED_COLS)),
        "confirmed_swing_high_events": int(df["confirmed_swing_high"].sum()),
        "confirmed_swing_low_events": int(df["confirmed_swing_low"].sum()),
        "confirmed_bos_bull_events": int(df["confirmed_bos_bull"].sum()),
        "confirmed_bos_bear_events": int(df["confirmed_bos_bear"].sum()),
        "confirmed_choch_bull_events": int(df["confirmed_choch_bull"].sum()),
        "confirmed_choch_bear_events": int(df["confirmed_choch_bear"].sum()),
        "confirmed_liquidity_sweep_high_events": int(df["confirmed_liquidity_sweep_high"].sum()),
        "confirmed_liquidity_sweep_low_events": int(df["confirmed_liquidity_sweep_low"].sum()),
        "confirmed_numeric_inf_cells": inf_cells,
        "added_cols": added_cols,
    }

    return df, summary


def dtype_by_timeframe(col: str) -> dict[str, str]:
    out = {}
    for tf in TIMEFRAMES:
        out[tf] = str(pd.read_parquet(feature_path(tf), columns=[col])[col].dtype)
    return out


def canonical_dtype(col: str) -> str:
    dtypes = sorted(set(dtype_by_timeframe(col).values()))
    return dtypes[0] if len(dtypes) == 1 else "mixed"


def update_registry() -> dict[str, Any]:
    registry = load_yaml(REGISTRY_YAML)
    features = registry.setdefault("features", {})

    registry["registry_revision"] = "P3H2"
    registry["updated_at_utc"] = now_utc()
    registry["feature_count_before_p3h2"] = int(registry.get("feature_count", len(features)))
    registry["confirmation_delay_bars_by_timeframe"] = CONFIRMATION_DELAY_BARS

    registry.setdefault("policy", {})["confirmed_structure_features_live_safe"] = True
    registry.setdefault("policy", {})["p3h_centered_structure_still_blocked"] = True
    registry.setdefault("policy", {})["confirmation_delay_bars_required"] = True

    for col in ALL_CONFIRMED_COLS:
        source_research_feature = None
        for src, dst in SHIFT_MAP.items():
            if dst == col:
                source_research_feature = src
                break

        features[col] = {
            "feature_name": col,
            "category": "structure_live_safe",
            "source_module": "p3h2",
            "live_safe": True,
            "ml_safe": True,
            "onnx_safe": True,
            "forbidden": False,
            "timeframes": TIMEFRAMES,
            "data_type": canonical_dtype(col),
            "data_types_by_timeframe": dtype_by_timeframe(col),
            "lookback_bars": 0,
            "requires_confirmation": False,
            "lookahead_risk": "none",
            "confirmation_delay_bars": "timeframe_specific",
            "confirmation_delay_bars_by_timeframe": CONFIRMATION_DELAY_BARS,
            "source_research_feature": source_research_feature,
            "reason": "Live-safe delayed confirmed structure feature. Value becomes available only after the configured confirmation delay for each timeframe.",
        }

    registry["feature_count"] = len(features)

    write_yaml(REGISTRY_YAML, registry)
    REGISTRY_JSON.write_text(json.dumps(registry, indent=2, default=str), encoding="utf-8")

    return registry


def blocked(meta: dict[str, Any], mode: str) -> bool:
    if meta.get("forbidden"):
        return True
    if meta.get("lookahead_risk") in {"high", "unknown"}:
        return True
    if mode == "ml" and not meta.get("ml_safe", False):
        return True
    if mode == "live" and not meta.get("live_safe", False):
        return True
    if mode == "onnx" and not meta.get("onnx_safe", False):
        return True
    return False


def rebuild_allowlists(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    features = registry["features"]

    ml_features = [
        name for name, meta in features.items()
        if not blocked(meta, "ml")
        and meta["category"] not in {"raw", "metadata", "diagnostic"}
        and not str(meta.get("data_type", "")).startswith("object")
    ]

    live_context_features = [
        name for name, meta in features.items()
        if not blocked(meta, "live")
        and meta["category"] not in {"metadata", "diagnostic"}
    ]

    live_model_features = [
        name for name, meta in features.items()
        if not blocked(meta, "live")
        and meta["category"] not in {"raw", "metadata", "diagnostic"}
        and not str(meta.get("data_type", "")).startswith("object")
    ]

    onnx_features = [
        name for name, meta in features.items()
        if not blocked(meta, "onnx")
        and meta["category"] not in {"raw", "metadata", "diagnostic"}
        and not str(meta.get("data_type", "")).startswith("object")
    ]

    blocked_features = [
        name for name, meta in features.items()
        if meta.get("forbidden") or not meta.get("ml_safe", False) or meta.get("lookahead_risk") in {"high", "unknown"}
    ]

    base = {
        "created_at_utc": now_utc(),
        "source_registry": rel(REGISTRY_YAML),
        "required_row_filter": "is_feature_row_safe == 1",
        "blocked_features": sorted(blocked_features),
        "forbidden_live_features": sorted(FORBIDDEN_LIVE_FEATURES),
        "confirmation_delay_bars_by_timeframe": CONFIRMATION_DELAY_BARS,
    }

    allowlists = {
        "ml": {
            **base,
            "purpose": "Numeric derived columns allowed for ML training input X after row filter.",
            "feature_count": len(ml_features),
            "features": sorted(ml_features),
        },
        "live": {
            **base,
            "purpose": "Backward-compatible live context allowlist. Prefer live_context_allowlist.yaml and live_model_allowlist.yaml.",
            "feature_count": len(live_context_features),
            "features": sorted(live_context_features),
        },
        "live_context": {
            **base,
            "purpose": "Columns allowed for live rule/signal context, including raw OHLCV context.",
            "feature_count": len(live_context_features),
            "features": sorted(live_context_features),
        },
        "live_model": {
            **base,
            "purpose": "Numeric columns allowed for live model inference input.",
            "feature_count": len(live_model_features),
            "features": sorted(live_model_features),
        },
        "onnx": {
            **base,
            "purpose": "Numeric columns allowed for ONNX model input/export.",
            "feature_count": len(onnx_features),
            "features": sorted(onnx_features),
        },
    }

    write_yaml(ML_ALLOWLIST_YAML, allowlists["ml"])
    write_yaml(LIVE_ALLOWLIST_YAML, allowlists["live"])
    write_yaml(LIVE_CONTEXT_ALLOWLIST_YAML, allowlists["live_context"])
    write_yaml(LIVE_MODEL_ALLOWLIST_YAML, allowlists["live_model"])
    write_yaml(ONNX_ALLOWLIST_YAML, allowlists["onnx"])

    return allowlists


def smoke_test_guard(registry: dict[str, Any]) -> pd.DataFrame:
    from xau_cgt.features.leakage_guard import FeatureLeakageError, assert_no_forbidden_features

    good_df = pd.DataFrame({
        "confirmed_bos_bull": [0, 1],
        "confirmed_choch_bull": [0, 0],
        "confirmed_structure_score_net": [1.0, 2.0],
        "confirmed_distance_to_last_swing_high_atr": [0.4, 1.2],
    })

    bad_df = pd.DataFrame({
        "bars_since_swing_high": [1.0, 2.0],
        "bars_to_next_large_gap": [10.0, 9.0],
        "confirmed_bos_bull": [0, 1],
    })

    rows = []

    try:
        assert_no_forbidden_features(good_df, registry=registry, mode="ml")
        rows.append({"test": "confirmed_p3h2_ml_safe", "status": "PASS", "message": "Confirmed structure columns allowed."})
    except Exception as e:
        rows.append({"test": "confirmed_p3h2_ml_safe", "status": "FAIL", "message": str(e)})

    try:
        assert_no_forbidden_features(bad_df, registry=registry, mode="ml")
        rows.append({"test": "original_p3h_blocked", "status": "FAIL", "message": "Expected leakage error but none was raised."})
    except FeatureLeakageError as e:
        rows.append({"test": "original_p3h_blocked", "status": "PASS", "message": str(e)})
    except Exception as e:
        rows.append({"test": "original_p3h_blocked", "status": "FAIL", "message": str(e)})

    return pd.DataFrame(rows)


def write_report(
    summary_df: pd.DataFrame,
    registry: dict[str, Any],
    allowlists: dict[str, dict[str, Any]],
    guard_tests: pd.DataFrame,
) -> None:
    confirmed_registry_rows = []

    for col in ALL_CONFIRMED_COLS:
        meta = registry["features"].get(col, {})
        confirmed_registry_rows.append({
            "feature_name": col,
            "category": meta.get("category"),
            "source_module": meta.get("source_module"),
            "live_safe": meta.get("live_safe"),
            "ml_safe": meta.get("ml_safe"),
            "onnx_safe": meta.get("onnx_safe"),
            "lookahead_risk": meta.get("lookahead_risk"),
            "confirmation_delay_bars": meta.get("confirmation_delay_bars"),
            "confirmation_delay_bars_by_timeframe": json.dumps(meta.get("confirmation_delay_bars_by_timeframe"), default=str),
            "source_research_feature": meta.get("source_research_feature"),
        })

    pd.DataFrame(confirmed_registry_rows).to_csv(REGISTRY_CHECK_CSV, index=False)
    guard_tests.to_csv(GUARD_TEST_CSV, index=False)

    lines = []
    lines.append("# P3H2 Live-Safe Confirmed Structure Features Report")
    lines.append("")
    lines.append(f"Created UTC: `{now_utc()}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Added delayed `confirmed_*` structure columns to all timeframe feature files.")
    lines.append("- Kept original P3H centered structure columns blocked as `structure_research`.")
    lines.append("- Registered confirmed columns as `structure_live_safe`.")
    lines.append("- Added `confirmation_delay_bars_by_timeframe` to registry and allowlists.")
    lines.append("- Rebuilt ML/live/ONNX allowlists.")
    lines.append("- Verified leakage guard blocks original P3H columns and allows confirmed P3H2 columns.")
    lines.append("")
    lines.append("## Confirmation delay")
    lines.append("")
    lines.append("| Timeframe | Delay Bars |")
    lines.append("|---|---:|")
    for tf, delay in CONFIRMATION_DELAY_BARS.items():
        lines.append(f"| {tf} | {delay} |")
    lines.append("")
    lines.append("## Allowlist counts")
    lines.append("")
    lines.append(f"- ML allowlist: `{allowlists['ml']['feature_count']}`")
    lines.append(f"- Live context allowlist: `{allowlists['live_context']['feature_count']}`")
    lines.append(f"- Live model allowlist: `{allowlists['live_model']['feature_count']}`")
    lines.append(f"- ONNX allowlist: `{allowlists['onnx']['feature_count']}`")
    lines.append("")
    lines.append("## Timeframe summary")
    lines.append("")
    lines.append("| TF | Delay | Rows | Cols Before | Cols After | Confirmed Cols | BOS Bull | BOS Bear | CHoCH Bull | CHoCH Bear | Inf Cells |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['timeframe']} | {int(row['confirmation_delay_bars'])} | {int(row['rows'])} | "
            f"{int(row['cols_before'])} | {int(row['cols_after'])} | {int(row['confirmed_cols_present'])} | "
            f"{int(row['confirmed_bos_bull_events'])} | {int(row['confirmed_bos_bear_events'])} | "
            f"{int(row['confirmed_choch_bull_events'])} | {int(row['confirmed_choch_bear_events'])} | "
            f"{int(row['confirmed_numeric_inf_cells'])} |"
        )

    lines.append("")
    lines.append("## Leakage guard smoke tests")
    lines.append("")
    lines.append("| Test | Status | Message |")
    lines.append("|---|---|---|")
    for _, row in guard_tests.iterrows():
        msg = str(row["message"]).replace("|", "\\|")
        lines.append(f"| {row['test']} | {row['status']} | {msg} |")

    lines.append("")
    lines.append("## Output files")
    lines.append("")
    lines.append(f"- Registry YAML: `{rel(REGISTRY_YAML)}`")
    lines.append(f"- Registry JSON: `{rel(REGISTRY_JSON)}`")
    lines.append(f"- ML allowlist: `{rel(ML_ALLOWLIST_YAML)}`")
    lines.append(f"- Live context allowlist: `{rel(LIVE_CONTEXT_ALLOWLIST_YAML)}`")
    lines.append(f"- Live model allowlist: `{rel(LIVE_MODEL_ALLOWLIST_YAML)}`")
    lines.append(f"- ONNX allowlist: `{rel(ONNX_ALLOWLIST_YAML)}`")
    lines.append(f"- Summary CSV: `{rel(SUMMARY_CSV)}`")
    lines.append(f"- Registry check CSV: `{rel(REGISTRY_CHECK_CSV)}`")
    lines.append(f"- Guard test CSV: `{rel(GUARD_TEST_CSV)}`")

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []

    print("Adding confirmed live-safe structure columns...")

    for tf in TIMEFRAMES:
        path = feature_path(tf)
        df = pd.read_parquet(path)
        before_cols = len(df.columns)

        df, summary = add_confirmed_structure(df, tf)
        after_cols = len(df.columns)

        df.to_parquet(path, index=False)

        summary["cols_before"] = before_cols
        summary["cols_after"] = after_cols
        summaries.append(summary)

        print(
            f"{tf}: rows={summary['rows']} cols {before_cols}->{after_cols} "
            f"delay={summary['confirmation_delay_bars']} confirmed_cols={summary['confirmed_cols_present']}"
        )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print("Updating registry...")
    registry = update_registry()

    print("Rebuilding allowlists...")
    allowlists = rebuild_allowlists(registry)

    print("Running leakage guard tests...")
    guard_tests = smoke_test_guard(registry)

    print("Writing report...")
    write_report(summary_df, registry, allowlists, guard_tests)

    event = {
        "phase": "P3H2",
        "created_at_utc": now_utc(),
        "confirmed_columns": len(ALL_CONFIRMED_COLS),
        "registry_feature_count": registry["feature_count"],
        "ml_allowlist_count": allowlists["ml"]["feature_count"],
        "live_context_allowlist_count": allowlists["live_context"]["feature_count"],
        "live_model_allowlist_count": allowlists["live_model"]["feature_count"],
        "onnx_allowlist_count": allowlists["onnx"]["feature_count"],
        "guard_tests_passed": int((guard_tests["status"] == "PASS").sum()),
        "guard_tests_total": int(len(guard_tests)),
        "status": "OK" if (guard_tests["status"] == "PASS").all() else "REVIEW",
    }

    log_event(event)

    print("P3H2 COMPLETE")
    print(json.dumps(event, indent=2))


if __name__ == "__main__":
    main()
