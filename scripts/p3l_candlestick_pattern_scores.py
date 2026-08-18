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
MTF_ROOT = FEATURE_ROOT / "mtf_aligned"
REPORT_DIR = PROJECT_ROOT / "reports" / "data_quality"
LOG_DIR = PROJECT_ROOT / "logs" / "python"

REGISTRY_YAML = FEATURE_ROOT / "feature_registry_v2.yaml"
REGISTRY_JSON = FEATURE_ROOT / "feature_registry_v2.json"

ML_ALLOWLIST_YAML = FEATURE_ROOT / "ml_feature_allowlist.yaml"
LIVE_FEATURE_ALLOWLIST_YAML = FEATURE_ROOT / "live_feature_allowlist.yaml"
LIVE_CONTEXT_ALLOWLIST_YAML = FEATURE_ROOT / "live_context_allowlist.yaml"
LIVE_MODEL_ALLOWLIST_YAML = FEATURE_ROOT / "live_model_allowlist.yaml"
ONNX_ALLOWLIST_YAML = FEATURE_ROOT / "onnx_feature_allowlist.yaml"

REPORT_MD = REPORT_DIR / "p3l_candlestick_pattern_scores_report.md"
SUMMARY_CSV = REPORT_DIR / "p3l_candlestick_pattern_summary.csv"
REGISTRY_CHECK_CSV = REPORT_DIR / "p3l_registry_pattern_columns.csv"
GUARD_TEST_CSV = REPORT_DIR / "p3l_leakage_guard_smoke_test.csv"
LOG_JSONL = LOG_DIR / "p3l_candlestick_pattern_scores.jsonl"

MTF_DATASETS = ["M15", "M5", "M1"]

FORBIDDEN_LIVE_FEATURES = {
    "bars_to_next_large_gap",
    "pre_gap_risk_bars_remaining",
    "pre_gap_risk_score_linear",
    "pre_gap_risk_score_exp",
    "pre_gap_risk_score",
    "is_pre_large_gap_risk",
}

BASE_PATTERN_SCORE_COLUMNS = [
    "pin_bar_bull_score",
    "pin_bar_bear_score",
    "engulfing_bull_score",
    "engulfing_bear_score",
    "inside_bar_score",
    "doji_score",
    "hammer_score",
    "shooting_star_score",
    "impulse_candle_bull_score",
    "impulse_candle_bear_score",
    "wick_rejection_high_score",
    "wick_rejection_low_score",
]

AGGREGATE_PATTERN_COLUMNS = [
    "pattern_bull_score",
    "pattern_bear_score",
    "pattern_indecision_score",
    "pattern_net_score",
    "pattern_confidence_score",
]

MODEL_SAFE_PATTERN_COLUMNS = BASE_PATTERN_SCORE_COLUMNS + AGGREGATE_PATTERN_COLUMNS

DIAGNOSTIC_PATTERN_COLUMNS = [
    "pattern_row_safe",
    "pattern_safety_reason",
]

ALL_PATTERN_COLUMNS = MODEL_SAFE_PATTERN_COLUMNS + DIAGNOSTIC_PATTERN_COLUMNS


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def log_event(record: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def mtf_path(base_tf: str) -> Path:
    return MTF_ROOT / f"base_timeframe={base_tf}" / f"xauusd_{base_tf}_mtf_features.parquet"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def clip01(x: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return np.clip(x, 0.0, 1.0)


def require_columns(df: pd.DataFrame, cols: list[str], dataset_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name}: missing required columns: {missing}")


def safe_bool(df: pd.DataFrame, col: str, default: int = 0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(bool(default), index=df.index)
    return df[col].fillna(default).astype(bool)


def safe_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float32")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def direction_context_multiplier(df: pd.DataFrame, direction: str) -> pd.Series:
    """
    Context multiplier is deliberately modest.
    Good context improves score, bad context reduces it, but geometry still matters.
    """
    if direction not in {"bull", "bear"}:
        raise ValueError("direction must be bull or bear")

    n = len(df)
    mult = pd.Series(1.0, index=df.index, dtype="float32")

    base_trend = safe_num(df, "confirmed_structure_trend_code", 0.0)
    mtf_alignment = safe_num(df, "mtf_structure_alignment_score", 0.0)
    mtf_signal = safe_num(df, "mtf_structure_signal_score", 0.0)

    near_base_low = safe_bool(df, "confirmed_near_last_swing_low", 0)
    near_base_high = safe_bool(df, "confirmed_near_last_swing_high", 0)
    near_mtf_support = safe_bool(df, "mtf_near_higher_tf_support", 0)
    near_mtf_resistance = safe_bool(df, "mtf_near_higher_tf_resistance", 0)

    if direction == "bull":
        mult += np.where(base_trend > 0, 0.15, 0.0).astype("float32")
        mult += np.where(base_trend < 0, -0.12, 0.0).astype("float32")
        mult += np.where(mtf_alignment > 0, 0.15, 0.0).astype("float32")
        mult += np.where(mtf_alignment < 0, -0.12, 0.0).astype("float32")
        mult += np.where(mtf_signal > 0, 0.08, 0.0).astype("float32")
        mult += np.where(near_base_low, 0.18, 0.0).astype("float32")
        mult += np.where(near_mtf_support, 0.22, 0.0).astype("float32")
        mult += np.where(near_mtf_resistance, -0.10, 0.0).astype("float32")
    else:
        mult += np.where(base_trend < 0, 0.15, 0.0).astype("float32")
        mult += np.where(base_trend > 0, -0.12, 0.0).astype("float32")
        mult += np.where(mtf_alignment < 0, 0.15, 0.0).astype("float32")
        mult += np.where(mtf_alignment > 0, -0.12, 0.0).astype("float32")
        mult += np.where(mtf_signal < 0, 0.08, 0.0).astype("float32")
        mult += np.where(near_base_high, 0.18, 0.0).astype("float32")
        mult += np.where(near_mtf_resistance, 0.22, 0.0).astype("float32")
        mult += np.where(near_mtf_support, -0.10, 0.0).astype("float32")

    return pd.Series(np.clip(mult.to_numpy(), 0.60, 1.50), index=df.index, dtype="float32")


def compute_patterns(df: pd.DataFrame, base_tf: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    before_cols = len(df.columns)

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "atr_14",
        "is_feature_row_safe",
        "confirmed_structure_row_safe",
        "mtf_context_row_safe",
        "mtf_confluence_row_safe",
    ]
    require_columns(df, required, f"P3L base={base_tf}")

    open_ = pd.to_numeric(df["open"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    atr = pd.to_numeric(df["atr_14"], errors="coerce").replace(0, np.nan)

    prev_open = open_.shift(1)
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    candle_range = (high - low).replace(0, np.nan)
    body = (close - open_).abs()
    signed_body = close - open_

    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low

    body_ratio = (body / candle_range).replace([np.inf, -np.inf], np.nan)
    upper_wick_ratio = (upper_wick / candle_range).replace([np.inf, -np.inf], np.nan)
    lower_wick_ratio = (lower_wick / candle_range).replace([np.inf, -np.inf], np.nan)
    range_atr = (candle_range / atr).replace([np.inf, -np.inf], np.nan)
    body_atr = (body / atr).replace([np.inf, -np.inf], np.nan)

    bullish_candle = close > open_
    bearish_candle = close < open_

    bull_mult = direction_context_multiplier(df, "bull")
    bear_mult = direction_context_multiplier(df, "bear")

    # Base geometry strengths.
    bull_pin_base = (
        clip01((lower_wick_ratio - 0.45) / 0.35)
        * clip01((0.35 - body_ratio) / 0.25)
        * clip01((0.25 - upper_wick_ratio) / 0.25)
    )

    bear_pin_base = (
        clip01((upper_wick_ratio - 0.45) / 0.35)
        * clip01((0.35 - body_ratio) / 0.25)
        * clip01((0.25 - lower_wick_ratio) / 0.25)
    )

    prev_body_high = np.maximum(prev_open, prev_close)
    prev_body_low = np.minimum(prev_open, prev_close)
    curr_body_high = np.maximum(open_, close)
    curr_body_low = np.minimum(open_, close)

    prev_body = (prev_close - prev_open).abs().replace(0, np.nan)

    engulf_bull_geom = (
        bullish_candle
        & (prev_close < prev_open)
        & (curr_body_high >= prev_body_high)
        & (curr_body_low <= prev_body_low)
    )
    engulf_bear_geom = (
        bearish_candle
        & (prev_close > prev_open)
        & (curr_body_high >= prev_body_high)
        & (curr_body_low <= prev_body_low)
    )

    engulf_strength = clip01((body / prev_body).replace([np.inf, -np.inf], np.nan) / 1.50).fillna(0.0)

    inside_bar_geom = (high <= prev_high) & (low >= prev_low)
    inside_base = (
        inside_bar_geom.astype("float32")
        * clip01((0.60 - body_ratio) / 0.50).fillna(0.0)
        * clip01((1.50 - range_atr) / 1.50).fillna(0.0)
    )

    doji_base = (
        clip01((0.18 - body_ratio) / 0.18).fillna(0.0)
        * clip01(range_atr / 0.75).fillna(0.0)
    )

    impulse_bull_base = (
        bullish_candle.astype("float32")
        * clip01((body_atr - 0.45) / 1.25).fillna(0.0)
        * clip01((body_ratio - 0.55) / 0.35).fillna(0.0)
    )

    impulse_bear_base = (
        bearish_candle.astype("float32")
        * clip01((body_atr - 0.45) / 1.25).fillna(0.0)
        * clip01((body_ratio - 0.55) / 0.35).fillna(0.0)
    )

    wick_rejection_low_base = (
        clip01((lower_wick_ratio - 0.35) / 0.45).fillna(0.0)
        * clip01(range_atr / 1.00).fillna(0.0)
    )

    wick_rejection_high_base = (
        clip01((upper_wick_ratio - 0.35) / 0.45).fillna(0.0)
        * clip01(range_atr / 1.00).fillna(0.0)
    )

    atr_body_mult = pd.Series(np.clip(0.85 + 0.25 * clip01(body_atr / 1.0), 0.85, 1.10), index=df.index, dtype="float32")
    atr_range_mult = pd.Series(np.clip(0.85 + 0.20 * clip01(range_atr / 1.2), 0.85, 1.08), index=df.index, dtype="float32")

    df["pin_bar_bull_score"] = clip01(bull_pin_base * bull_mult * atr_range_mult).astype("float32")
    df["pin_bar_bear_score"] = clip01(bear_pin_base * bear_mult * atr_range_mult).astype("float32")

    df["hammer_score"] = clip01(df["pin_bar_bull_score"] * (1.0 + 0.15 * safe_bool(df, "confirmed_near_last_swing_low", 0).astype("float32"))).astype("float32")
    df["shooting_star_score"] = clip01(df["pin_bar_bear_score"] * (1.0 + 0.15 * safe_bool(df, "confirmed_near_last_swing_high", 0).astype("float32"))).astype("float32")

    df["engulfing_bull_score"] = clip01(engulf_bull_geom.astype("float32") * engulf_strength * bull_mult * atr_body_mult).astype("float32")
    df["engulfing_bear_score"] = clip01(engulf_bear_geom.astype("float32") * engulf_strength * bear_mult * atr_body_mult).astype("float32")

    # Inside bar is not directional. Boost if near confluence because breakout setup matters more around levels.
    confluence_presence = (
        safe_bool(df, "mtf_near_higher_tf_support", 0).astype("float32")
        + safe_bool(df, "mtf_near_higher_tf_resistance", 0).astype("float32")
    ).clip(0, 1)
    df["inside_bar_score"] = clip01(inside_base * (1.0 + 0.20 * confluence_presence)).astype("float32")

    # Doji/indecision gets boosted near HTF support/resistance and when structure is conflicted.
    conflict = safe_num(df, "mtf_structure_conflict_score", 0.0)
    conflict_mult = pd.Series(np.clip(1.0 + 0.12 * conflict, 1.0, 1.30), index=df.index, dtype="float32")
    df["doji_score"] = clip01(doji_base * (1.0 + 0.18 * confluence_presence) * conflict_mult).astype("float32")

    df["impulse_candle_bull_score"] = clip01(impulse_bull_base * bull_mult).astype("float32")
    df["impulse_candle_bear_score"] = clip01(impulse_bear_base * bear_mult).astype("float32")

    df["wick_rejection_low_score"] = clip01(wick_rejection_low_base * bull_mult).astype("float32")
    df["wick_rejection_high_score"] = clip01(wick_rejection_high_base * bear_mult).astype("float32")

    bull_cols = [
        "pin_bar_bull_score",
        "hammer_score",
        "engulfing_bull_score",
        "impulse_candle_bull_score",
        "wick_rejection_low_score",
    ]
    bear_cols = [
        "pin_bar_bear_score",
        "shooting_star_score",
        "engulfing_bear_score",
        "impulse_candle_bear_score",
        "wick_rejection_high_score",
    ]

    df["pattern_bull_score"] = df[bull_cols].max(axis=1).astype("float32")
    df["pattern_bear_score"] = df[bear_cols].max(axis=1).astype("float32")
    df["pattern_indecision_score"] = df[["inside_bar_score", "doji_score"]].max(axis=1).astype("float32")
    df["pattern_net_score"] = (df["pattern_bull_score"] - df["pattern_bear_score"]).astype("float32")
    df["pattern_confidence_score"] = df[["pattern_bull_score", "pattern_bear_score", "pattern_indecision_score"]].max(axis=1).astype("float32")

    # Replace numeric infinities, preserve NaNs only in intermediate not model outputs.
    for col in MODEL_SAFE_PATTERN_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")

    base_safe = safe_bool(df, "is_feature_row_safe", 0)
    structure_safe = safe_bool(df, "confirmed_structure_row_safe", 0)
    mtf_safe = safe_bool(df, "mtf_context_row_safe", 0)
    confluence_safe = safe_bool(df, "mtf_confluence_row_safe", 0)

    core_present = pd.Series(True, index=df.index)
    for col in MODEL_SAFE_PATTERN_COLUMNS:
        core_present &= df[col].notna()

    safe = base_safe & structure_safe & mtf_safe & confluence_safe & core_present

    reason = np.full(len(df), "SAFE", dtype=object)
    reason[~base_safe.to_numpy()] = "BASE_FEATURE_ROW_UNSAFE"
    reason[(base_safe & ~structure_safe).to_numpy()] = "BASE_STRUCTURE_UNSAFE"
    reason[(base_safe & structure_safe & ~mtf_safe).to_numpy()] = "MTF_CONTEXT_UNSAFE"
    reason[(base_safe & structure_safe & mtf_safe & ~confluence_safe).to_numpy()] = "MTF_CONFLUENCE_UNSAFE"
    reason[(base_safe & structure_safe & mtf_safe & confluence_safe & ~core_present).to_numpy()] = "PATTERN_CORE_MISSING"

    df["pattern_row_safe"] = safe.astype("int8")
    df["pattern_safety_reason"] = pd.Series(reason, index=df.index).astype("string")

    numeric_inf_cells = int(np.isinf(df[MODEL_SAFE_PATTERN_COLUMNS].select_dtypes(include=[np.number]).to_numpy()).sum())

    summary = {
        "base_timeframe": base_tf,
        "rows": int(len(df)),
        "cols_before": int(before_cols),
        "cols_after": int(len(df.columns)),
        "cols_added": int(len(df.columns) - before_cols),
        "pattern_safe_rows": int(df["pattern_row_safe"].sum()),
        "pattern_safe_ratio": float(df["pattern_row_safe"].mean()),
        "pin_bar_bull_active_rows": int((df["pin_bar_bull_score"] > 0.50).sum()),
        "pin_bar_bear_active_rows": int((df["pin_bar_bear_score"] > 0.50).sum()),
        "engulfing_bull_active_rows": int((df["engulfing_bull_score"] > 0.50).sum()),
        "engulfing_bear_active_rows": int((df["engulfing_bear_score"] > 0.50).sum()),
        "inside_bar_active_rows": int((df["inside_bar_score"] > 0.50).sum()),
        "doji_active_rows": int((df["doji_score"] > 0.50).sum()),
        "impulse_bull_active_rows": int((df["impulse_candle_bull_score"] > 0.50).sum()),
        "impulse_bear_active_rows": int((df["impulse_candle_bear_score"] > 0.50).sum()),
        "wick_rejection_low_active_rows": int((df["wick_rejection_low_score"] > 0.50).sum()),
        "wick_rejection_high_active_rows": int((df["wick_rejection_high_score"] > 0.50).sum()),
        "bull_score_mean": float(df["pattern_bull_score"].mean()),
        "bear_score_mean": float(df["pattern_bear_score"].mean()),
        "indecision_score_mean": float(df["pattern_indecision_score"].mean()),
        "confidence_score_mean": float(df["pattern_confidence_score"].mean()),
        "numeric_inf_cells": numeric_inf_cells,
    }

    return df, summary


def update_registry() -> dict[str, Any]:
    registry = load_yaml(REGISTRY_YAML)
    features = registry.setdefault("features", {})

    registry["registry_revision"] = "P3L"
    registry["updated_at_utc"] = now_utc()
    registry.setdefault("policy", {})["candlestick_pattern_scores_enabled"] = True
    registry.setdefault("policy", {})["pattern_row_safe_required"] = True
    registry.setdefault("policy", {})["pattern_scores_confirmation_delay_bars"] = 1

    sample_path = mtf_path("M15")
    sample_df = pd.read_parquet(sample_path, columns=MODEL_SAFE_PATTERN_COLUMNS)

    for col in MODEL_SAFE_PATTERN_COLUMNS:
        features[col] = {
            "feature_name": col,
            "category": "pattern",
            "source_module": "p3l",
            "live_safe": True,
            "ml_safe": True,
            "onnx_safe": True,
            "forbidden": False,
            "base_timeframes": MTF_DATASETS,
            "data_type": str(sample_df[col].dtype),
            "lookahead_risk": "none",
            "requires_confirmation": True,
            "confirmation_delay_bars": 1,
            "depends_on_filter": "mtf_confluence_row_safe == 1",
            "filter_required_for_use": "pattern_row_safe == 1",
            "score_range": "0.0_to_1.0_or_signed_for_net_score",
            "reason": "Live-safe candlestick pattern score computed from closed candle geometry and structure/MTF/confluence context.",
        }

    features["pattern_row_safe"] = {
        "feature_name": "pattern_row_safe",
        "category": "diagnostic",
        "source_module": "p3l",
        "live_safe": False,
        "ml_safe": False,
        "onnx_safe": False,
        "forbidden": False,
        "base_timeframes": MTF_DATASETS,
        "data_type": "int8",
        "lookahead_risk": "none",
        "filter_role": "required_filter_for_pattern_features",
        "reason": "Filter/diagnostic column. Use to gate P3L pattern features; do not feed into model X.",
    }

    features["pattern_safety_reason"] = {
        "feature_name": "pattern_safety_reason",
        "category": "diagnostic",
        "source_module": "p3l",
        "live_safe": False,
        "ml_safe": False,
        "onnx_safe": False,
        "forbidden": False,
        "base_timeframes": MTF_DATASETS,
        "data_type": "string",
        "lookahead_risk": "none",
        "filter_role": "audit_reason_for_pattern_row_safe",
        "reason": "Diagnostic reason for P3L pattern safety. Not a model feature.",
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


def is_numeric_model_candidate(meta: dict[str, Any]) -> bool:
    dtype = str(meta.get("data_type", ""))
    return not (
        dtype.startswith("object")
        or dtype.startswith("string")
        or dtype.startswith("datetime")
        or dtype.startswith("category")
    )


def rebuild_allowlists(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    features = registry["features"]

    blocked_categories = {
        "raw",
        "metadata",
        "diagnostic",
        "mtf_metadata",
        "mtf_filter",
    }

    ml_features = [
        name for name, meta in features.items()
        if not blocked(meta, "ml")
        and meta.get("category") not in blocked_categories
        and is_numeric_model_candidate(meta)
    ]

    live_context_features = [
        name for name, meta in features.items()
        if not blocked(meta, "live")
        and meta.get("category") not in {"metadata", "diagnostic", "mtf_metadata", "mtf_filter"}
    ]

    live_model_features = [
        name for name, meta in features.items()
        if not blocked(meta, "live")
        and meta.get("category") not in blocked_categories
        and is_numeric_model_candidate(meta)
    ]

    onnx_features = [
        name for name, meta in features.items()
        if not blocked(meta, "onnx")
        and meta.get("category") not in blocked_categories
        and is_numeric_model_candidate(meta)
    ]

    blocked_features = [
        name for name, meta in features.items()
        if meta.get("forbidden") or not meta.get("ml_safe", False) or meta.get("lookahead_risk") in {"high", "unknown"}
    ]

    base = {
        "created_at_utc": now_utc(),
        "source_registry": rel(REGISTRY_YAML),
        "required_row_filter": "is_feature_row_safe == 1",
        "required_confirmed_structure_filter": "confirmed_structure_row_safe == 1",
        "required_mtf_context_filter": "mtf_context_row_safe == 1",
        "required_mtf_confluence_filter": "mtf_confluence_row_safe == 1",
        "required_pattern_filter": "pattern_row_safe == 1",
        "blocked_features": sorted(blocked_features),
        "forbidden_live_features": sorted(FORBIDDEN_LIVE_FEATURES),
    }

    allowlists = {
        "ml": {
            **base,
            "purpose": "Numeric derived columns allowed for ML training input X after all P3 safety filters.",
            "feature_count": len(ml_features),
            "features": sorted(ml_features),
        },
        "live_context": {
            **base,
            "purpose": "Columns allowed for live rule/signal context.",
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
    write_yaml(LIVE_CONTEXT_ALLOWLIST_YAML, allowlists["live_context"])
    write_yaml(LIVE_MODEL_ALLOWLIST_YAML, allowlists["live_model"])
    write_yaml(ONNX_ALLOWLIST_YAML, allowlists["onnx"])
    write_yaml(LIVE_FEATURE_ALLOWLIST_YAML, allowlists["live_context"])

    return allowlists


def smoke_test_guard(registry: dict[str, Any]) -> pd.DataFrame:
    from xau_cgt.features.leakage_guard import FeatureLeakageError, assert_no_forbidden_features

    good_df = pd.DataFrame({
        "pin_bar_bull_score": [0.1, 0.8],
        "engulfing_bear_score": [0.0, 0.6],
        "pattern_net_score": [0.1, -0.4],
        "pattern_confidence_score": [0.1, 0.8],
    })

    bad_df = pd.DataFrame({
        "pin_bar_bull_score": [0.1, 0.8],
        "pattern_row_safe": [1, 1],
        "pattern_safety_reason": ["SAFE", "SAFE"],
        "bars_to_next_large_gap": [5.0, 6.0],
    })

    rows = []

    try:
        assert_no_forbidden_features(good_df, registry=registry, mode="ml")
        rows.append({"test": "p3l_pattern_model_features_allowed", "status": "PASS", "message": "P3L pattern model features allowed."})
    except Exception as e:
        rows.append({"test": "p3l_pattern_model_features_allowed", "status": "FAIL", "message": str(e)})

    try:
        assert_no_forbidden_features(bad_df, registry=registry, mode="ml")
        rows.append({"test": "p3l_diagnostics_blocked_from_ml", "status": "FAIL", "message": "Expected leakage error but none was raised."})
    except FeatureLeakageError as e:
        rows.append({"test": "p3l_diagnostics_blocked_from_ml", "status": "PASS", "message": str(e)})
    except Exception as e:
        rows.append({"test": "p3l_diagnostics_blocked_from_ml", "status": "FAIL", "message": str(e)})

    return pd.DataFrame(rows)


def write_report(
    summary_df: pd.DataFrame,
    registry: dict[str, Any],
    allowlists: dict[str, dict[str, Any]],
    guard_tests: pd.DataFrame,
) -> None:
    registry_rows = []
    for col in ALL_PATTERN_COLUMNS:
        meta = registry["features"].get(col, {})
        registry_rows.append({
            "feature_name": col,
            "category": meta.get("category"),
            "source_module": meta.get("source_module"),
            "live_safe": meta.get("live_safe"),
            "ml_safe": meta.get("ml_safe"),
            "onnx_safe": meta.get("onnx_safe"),
            "lookahead_risk": meta.get("lookahead_risk"),
            "data_type": meta.get("data_type"),
            "confirmation_delay_bars": meta.get("confirmation_delay_bars"),
            "filter_role": meta.get("filter_role"),
            "filter_required_for_use": meta.get("filter_required_for_use"),
        })

    pd.DataFrame(registry_rows).to_csv(REGISTRY_CHECK_CSV, index=False)
    guard_tests.to_csv(GUARD_TEST_CSV, index=False)

    lines = []
    lines.append("# P3L Candlestick Pattern Scores Report")
    lines.append("")
    lines.append(f"Created UTC: `{now_utc()}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Added context-aware candlestick pattern scores.")
    lines.append("- Built 12 base pattern scores plus 5 aggregate scores.")
    lines.append("- Scores are capped numeric values, not binary signals.")
    lines.append("- Pattern scores use candle geometry, ATR context, structure alignment, MTF alignment, and confluence zones.")
    lines.append("- Added `pattern_row_safe` and `pattern_safety_reason` as diagnostic/filter-only columns.")
    lines.append("- Registered model-safe P3L columns as `pattern`.")
    lines.append("- Rebuilt ML/live/ONNX allowlists.")
    lines.append("")
    lines.append("## Required filters")
    lines.append("")
    lines.append("```python")
    lines.append("df = df[df['is_feature_row_safe'] == 1]")
    lines.append("df = df[df['confirmed_structure_row_safe'] == 1]")
    lines.append("df = df[df['mtf_context_row_safe'] == 1]")
    lines.append("df = df[df['mtf_confluence_row_safe'] == 1]")
    lines.append("df = df[df['pattern_row_safe'] == 1]")
    lines.append("```")
    lines.append("")
    lines.append("## Pattern columns")
    lines.append("")
    lines.append("| Column | Model Safe |")
    lines.append("|---|---:|")
    for col in MODEL_SAFE_PATTERN_COLUMNS:
        lines.append(f"| {col} | 1 |")
    for col in DIAGNOSTIC_PATTERN_COLUMNS:
        lines.append(f"| {col} | 0 |")
    lines.append("")
    lines.append("## Dataset summary")
    lines.append("")
    lines.append("| Base TF | Rows | Cols Before | Cols After | Safe Rows | Safe Ratio | Pin Bull >0.5 | Pin Bear >0.5 | Engulf Bull >0.5 | Engulf Bear >0.5 | Inside >0.5 | Doji >0.5 | Impulse Bull >0.5 | Impulse Bear >0.5 | Inf Cells |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['base_timeframe']} | {int(row['rows'])} | {int(row['cols_before'])} | {int(row['cols_after'])} | "
            f"{int(row['pattern_safe_rows'])} | {float(row['pattern_safe_ratio']):.6f} | "
            f"{int(row['pin_bar_bull_active_rows'])} | {int(row['pin_bar_bear_active_rows'])} | "
            f"{int(row['engulfing_bull_active_rows'])} | {int(row['engulfing_bear_active_rows'])} | "
            f"{int(row['inside_bar_active_rows'])} | {int(row['doji_active_rows'])} | "
            f"{int(row['impulse_bull_active_rows'])} | {int(row['impulse_bear_active_rows'])} | "
            f"{int(row['numeric_inf_cells'])} |"
        )
    lines.append("")
    lines.append("## Allowlist counts")
    lines.append("")
    lines.append(f"- ML allowlist: `{allowlists['ml']['feature_count']}`")
    lines.append(f"- Live context allowlist: `{allowlists['live_context']['feature_count']}`")
    lines.append(f"- Live model allowlist: `{allowlists['live_model']['feature_count']}`")
    lines.append(f"- ONNX allowlist: `{allowlists['onnx']['feature_count']}`")
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
    lines.append(f"- Summary CSV: `{rel(SUMMARY_CSV)}`")
    lines.append(f"- Registry check CSV: `{rel(REGISTRY_CHECK_CSV)}`")
    lines.append(f"- Guard test CSV: `{rel(GUARD_TEST_CSV)}`")
    lines.append(f"- Registry YAML: `{rel(REGISTRY_YAML)}`")

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []

    print("Adding P3L candlestick pattern scores...")

    for base_tf in MTF_DATASETS:
        path = mtf_path(base_tf)
        if not path.exists():
            raise FileNotFoundError(f"Missing MTF dataset: {path}")

        df = pd.read_parquet(path)
        df, summary = compute_patterns(df, base_tf)
        df.to_parquet(path, index=False)

        summaries.append(summary)

        print(
            f"{base_tf}: safe={summary['pattern_safe_rows']}/{summary['rows']} "
            f"ratio={summary['pattern_safe_ratio']:.4f} cols={summary['cols_before']}->{summary['cols_after']}"
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
        "phase": "P3L",
        "created_at_utc": now_utc(),
        "datasets_patched": len(summary_df),
        "p3l_model_features_added": len(MODEL_SAFE_PATTERN_COLUMNS),
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

    print("P3L COMPLETE")
    print(json.dumps(event, indent=2))


if __name__ == "__main__":
    main()
