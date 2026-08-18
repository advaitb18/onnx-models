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

REPORT_MD = REPORT_DIR / "p3k_mtf_swing_confluence_zones_report.md"
SUMMARY_CSV = REPORT_DIR / "p3k_mtf_swing_confluence_summary.csv"
REGISTRY_CHECK_CSV = REPORT_DIR / "p3k_registry_confluence_columns.csv"
GUARD_TEST_CSV = REPORT_DIR / "p3k_leakage_guard_smoke_test.csv"
LOG_JSONL = LOG_DIR / "p3k_mtf_swing_confluence_zones.jsonl"

MTF_JOBS = {
    "M15": ["H1", "H4", "D1"],
    "M5": ["M15", "H1", "H4"],
    "M1": ["M5", "M15", "H1"],
}

FORBIDDEN_LIVE_FEATURES = {
    "bars_to_next_large_gap",
    "pre_gap_risk_bars_remaining",
    "pre_gap_risk_score_linear",
    "pre_gap_risk_score_exp",
    "pre_gap_risk_score",
    "is_pre_large_gap_risk",
}

P3K_FEATURES = [
    "mtf_swing_high_confluence_count",
    "mtf_swing_low_confluence_count",
    "mtf_near_higher_tf_resistance",
    "mtf_near_higher_tf_support",
    "mtf_nearest_resistance_distance",
    "mtf_nearest_support_distance",
    "mtf_nearest_resistance_distance_atr",
    "mtf_nearest_support_distance_atr",
    "mtf_resistance_cluster_score",
    "mtf_support_cluster_score",
    "mtf_bullish_structure_votes",
    "mtf_bearish_structure_votes",
    "mtf_range_structure_votes",
    "mtf_structure_alignment_score",
    "mtf_structure_alignment_direction",
    "mtf_structure_conflict_score",
    "mtf_bos_bull_votes",
    "mtf_bos_bear_votes",
    "mtf_choch_bull_votes",
    "mtf_choch_bear_votes",
    "mtf_structure_signal_score",
    "mtf_confluence_row_safe",
    "mtf_confluence_safety_reason",
]

MODEL_SAFE_P3K_FEATURES = [
    c for c in P3K_FEATURES
    if c not in {"mtf_confluence_row_safe", "mtf_confluence_safety_reason"}
]


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


def require_columns(df: pd.DataFrame, cols: list[str], dataset_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name}: missing required columns: {missing}")


def compute_confluence(df: pd.DataFrame, base_tf: str, context_tfs: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    before_cols = len(df.columns)

    require_columns(
        df,
        [
            "close",
            "atr_14",
            "mtf_context_row_safe",
            "is_feature_row_safe",
            "confirmed_structure_row_safe",
        ],
        f"base={base_tf}",
    )

    close = pd.to_numeric(df["close"], errors="coerce")
    atr = pd.to_numeric(df["atr_14"], errors="coerce").replace(0, np.nan)

    resistance_count = pd.Series(0, index=df.index, dtype="int16")
    support_count = pd.Series(0, index=df.index, dtype="int16")

    resistance_cluster_score = pd.Series(0.0, index=df.index, dtype="float32")
    support_cluster_score = pd.Series(0.0, index=df.index, dtype="float32")

    nearest_resistance_distance = pd.Series(np.nan, index=df.index, dtype="float32")
    nearest_support_distance = pd.Series(np.nan, index=df.index, dtype="float32")

    bullish_votes = pd.Series(0, index=df.index, dtype="int16")
    bearish_votes = pd.Series(0, index=df.index, dtype="int16")
    range_votes = pd.Series(0, index=df.index, dtype="int16")

    bos_bull_votes = pd.Series(0, index=df.index, dtype="int16")
    bos_bear_votes = pd.Series(0, index=df.index, dtype="int16")
    choch_bull_votes = pd.Series(0, index=df.index, dtype="int16")
    choch_bear_votes = pd.Series(0, index=df.index, dtype="int16")

    context_valid_all = pd.Series(True, index=df.index)

    for ctx in context_tfs:
        required = [
            f"{ctx}_confirmed_last_swing_high",
            f"{ctx}_confirmed_last_swing_low",
            f"{ctx}_confirmed_structure_trend_code",
            f"{ctx}_confirmed_bos_bull",
            f"{ctx}_confirmed_bos_bear",
            f"{ctx}_confirmed_choch_bull",
            f"{ctx}_confirmed_choch_bear",
            f"{ctx}_context_fresh",
            f"{ctx}_context_feature_safe",
            f"{ctx}_context_confirmed_structure_safe",
        ]
        require_columns(df, required, f"base={base_tf}, context={ctx}")

        ctx_fresh = df[f"{ctx}_context_fresh"].fillna(0).astype(bool)
        ctx_feature_safe = df[f"{ctx}_context_feature_safe"].fillna(0).astype(bool)
        ctx_structure_safe = df[f"{ctx}_context_confirmed_structure_safe"].fillna(0).astype(bool)
        ctx_valid = ctx_fresh & ctx_feature_safe & ctx_structure_safe
        context_valid_all &= ctx_valid

        high_level = pd.to_numeric(df[f"{ctx}_confirmed_last_swing_high"], errors="coerce")
        low_level = pd.to_numeric(df[f"{ctx}_confirmed_last_swing_low"], errors="coerce")

        high_dist = high_level - close
        low_dist = close - low_level

        high_dist_atr = high_dist / atr
        low_dist_atr = low_dist / atr

        # Resistance/support is considered near if within 1 ATR above/below current price.
        near_resistance = ctx_valid & high_dist.notna() & (high_dist >= 0) & (high_dist_atr.abs() <= 1.0)
        near_support = ctx_valid & low_dist.notna() & (low_dist >= 0) & (low_dist_atr.abs() <= 1.0)

        resistance_count += near_resistance.astype("int16")
        support_count += near_support.astype("int16")

        # Score closer levels more strongly: 1 ATR away = 0, at current price = 1.
        resistance_cluster_score += np.where(
            near_resistance,
            np.maximum(0.0, 1.0 - high_dist_atr.abs()),
            0.0,
        ).astype("float32")

        support_cluster_score += np.where(
            near_support,
            np.maximum(0.0, 1.0 - low_dist_atr.abs()),
            0.0,
        ).astype("float32")

        # Track nearest distances.
        nearest_resistance_distance = pd.Series(
            np.fmin(
                nearest_resistance_distance.fillna(np.inf).to_numpy(),
                high_dist.where(near_resistance, np.inf).fillna(np.inf).to_numpy(),
            ),
            index=df.index,
            dtype="float32",
        ).replace(np.inf, np.nan)

        nearest_support_distance = pd.Series(
            np.fmin(
                nearest_support_distance.fillna(np.inf).to_numpy(),
                low_dist.where(near_support, np.inf).fillna(np.inf).to_numpy(),
            ),
            index=df.index,
            dtype="float32",
        ).replace(np.inf, np.nan)

        trend_code = pd.to_numeric(df[f"{ctx}_confirmed_structure_trend_code"], errors="coerce").fillna(0)
        bullish_votes += ((trend_code > 0) & ctx_valid).astype("int16")
        bearish_votes += ((trend_code < 0) & ctx_valid).astype("int16")
        range_votes += ((trend_code == 0) & ctx_valid).astype("int16")

        bos_bull_votes += (df[f"{ctx}_confirmed_bos_bull"].fillna(0).astype(bool) & ctx_valid).astype("int16")
        bos_bear_votes += (df[f"{ctx}_confirmed_bos_bear"].fillna(0).astype(bool) & ctx_valid).astype("int16")
        choch_bull_votes += (df[f"{ctx}_confirmed_choch_bull"].fillna(0).astype(bool) & ctx_valid).astype("int16")
        choch_bear_votes += (df[f"{ctx}_confirmed_choch_bear"].fillna(0).astype(bool) & ctx_valid).astype("int16")

    df["mtf_swing_high_confluence_count"] = resistance_count.astype("int8")
    df["mtf_swing_low_confluence_count"] = support_count.astype("int8")
    df["mtf_near_higher_tf_resistance"] = (resistance_count > 0).astype("int8")
    df["mtf_near_higher_tf_support"] = (support_count > 0).astype("int8")

    df["mtf_nearest_resistance_distance"] = nearest_resistance_distance.astype("float32")
    df["mtf_nearest_support_distance"] = nearest_support_distance.astype("float32")
    df["mtf_nearest_resistance_distance_atr"] = (nearest_resistance_distance / atr).astype("float32")
    df["mtf_nearest_support_distance_atr"] = (nearest_support_distance / atr).astype("float32")

    df["mtf_resistance_cluster_score"] = resistance_cluster_score.astype("float32")
    df["mtf_support_cluster_score"] = support_cluster_score.astype("float32")

    df["mtf_bullish_structure_votes"] = bullish_votes.astype("int8")
    df["mtf_bearish_structure_votes"] = bearish_votes.astype("int8")
    df["mtf_range_structure_votes"] = range_votes.astype("int8")

    alignment_score = bullish_votes - bearish_votes
    df["mtf_structure_alignment_score"] = alignment_score.astype("int8")
    df["mtf_structure_alignment_direction"] = np.select(
        [alignment_score > 0, alignment_score < 0],
        [1, -1],
        default=0,
    ).astype("int8")

    # Conflict when both bullish and bearish context timeframes vote at same row.
    df["mtf_structure_conflict_score"] = np.minimum(bullish_votes, bearish_votes).astype("int8")

    df["mtf_bos_bull_votes"] = bos_bull_votes.astype("int8")
    df["mtf_bos_bear_votes"] = bos_bear_votes.astype("int8")
    df["mtf_choch_bull_votes"] = choch_bull_votes.astype("int8")
    df["mtf_choch_bear_votes"] = choch_bear_votes.astype("int8")

    # Compact signal score: structure direction + break/choch impulse + support/resistance proximity.
    signal_score = (
        df["mtf_structure_alignment_score"].astype("float32")
        + 0.75 * (df["mtf_bos_bull_votes"].astype("float32") - df["mtf_bos_bear_votes"].astype("float32"))
        + 0.50 * (df["mtf_choch_bull_votes"].astype("float32") - df["mtf_choch_bear_votes"].astype("float32"))
        + 0.25 * (df["mtf_swing_low_confluence_count"].astype("float32") - df["mtf_swing_high_confluence_count"].astype("float32"))
    )
    df["mtf_structure_signal_score"] = signal_score.astype("float32")

    base_safe = df["is_feature_row_safe"].fillna(0).astype(bool)
    base_structure_safe = df["confirmed_structure_row_safe"].fillna(0).astype(bool)
    mtf_safe = df["mtf_context_row_safe"].fillna(0).astype(bool)

    core_present = pd.Series(True, index=df.index)
    for col in MODEL_SAFE_P3K_FEATURES:
        if col in {
            "mtf_nearest_resistance_distance",
            "mtf_nearest_support_distance",
            "mtf_nearest_resistance_distance_atr",
            "mtf_nearest_support_distance_atr",
        }:
            # Distance columns can be NaN if no nearby support/resistance exists.
            continue
        core_present &= df[col].notna()

    safe = base_safe & base_structure_safe & mtf_safe & context_valid_all & core_present

    reason = np.full(len(df), "SAFE", dtype=object)
    reason[~base_safe.to_numpy()] = "BASE_FEATURE_ROW_UNSAFE"
    reason[(base_safe & ~base_structure_safe).to_numpy()] = "BASE_STRUCTURE_UNSAFE"
    reason[(base_safe & base_structure_safe & ~mtf_safe).to_numpy()] = "MTF_CONTEXT_UNSAFE"
    reason[(base_safe & base_structure_safe & mtf_safe & ~context_valid_all).to_numpy()] = "CONTEXT_VALIDATION_FAILED"
    reason[(base_safe & base_structure_safe & mtf_safe & context_valid_all & ~core_present).to_numpy()] = "CONFLUENCE_CORE_MISSING"

    df["mtf_confluence_row_safe"] = safe.astype("int8")
    df["mtf_confluence_safety_reason"] = pd.Series(reason, index=df.index).astype("string")

    for col in MODEL_SAFE_P3K_FEATURES:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    summary = {
        "base_timeframe": base_tf,
        "context_timeframes": ",".join(context_tfs),
        "rows": int(len(df)),
        "cols_before": int(before_cols),
        "cols_after": int(len(df.columns)),
        "cols_added": int(len(df.columns) - before_cols),
        "mtf_context_safe_rows": int(df["mtf_context_row_safe"].sum()),
        "mtf_confluence_safe_rows": int(df["mtf_confluence_row_safe"].sum()),
        "mtf_confluence_safe_ratio": float(df["mtf_confluence_row_safe"].mean()),
        "near_resistance_rows": int(df["mtf_near_higher_tf_resistance"].sum()),
        "near_support_rows": int(df["mtf_near_higher_tf_support"].sum()),
        "bullish_vote_rows": int((df["mtf_bullish_structure_votes"] > 0).sum()),
        "bearish_vote_rows": int((df["mtf_bearish_structure_votes"] > 0).sum()),
        "conflict_rows": int((df["mtf_structure_conflict_score"] > 0).sum()),
        "bos_bull_vote_rows": int((df["mtf_bos_bull_votes"] > 0).sum()),
        "bos_bear_vote_rows": int((df["mtf_bos_bear_votes"] > 0).sum()),
        "numeric_inf_cells": int(
            np.isinf(df[MODEL_SAFE_P3K_FEATURES].select_dtypes(include=[np.number]).to_numpy()).sum()
        ),
    }

    return df, summary


def update_registry() -> dict[str, Any]:
    registry = load_yaml(REGISTRY_YAML)
    features = registry.setdefault("features", {})

    registry["registry_revision"] = "P3K"
    registry["updated_at_utc"] = now_utc()
    registry.setdefault("policy", {})["mtf_swing_confluence_features_enabled"] = True
    registry.setdefault("policy", {})["mtf_confluence_row_safe_required"] = True
    registry.setdefault("policy", {})["mtf_confluence_safety_columns_are_filters_not_model_features"] = True

    for col in MODEL_SAFE_P3K_FEATURES:
        # Infer dtype from M15 file; all P3K model columns are numeric and written consistently.
        sample_path = mtf_path("M15")
        dtype = str(pd.read_parquet(sample_path, columns=[col])[col].dtype)

        features[col] = {
            "feature_name": col,
            "category": "mtf_confluence",
            "source_module": "p3k",
            "live_safe": True,
            "ml_safe": True,
            "onnx_safe": True,
            "forbidden": False,
            "base_timeframes": list(MTF_JOBS.keys()),
            "data_type": dtype,
            "lookahead_risk": "none",
            "requires_confirmation": False,
            "depends_on_filter": "mtf_context_row_safe == 1",
            "filter_required_for_use": "mtf_confluence_row_safe == 1",
            "reason": "MTF swing/structure confluence feature computed from as-of aligned, stale-filtered higher timeframe confirmed structure context.",
        }

    features["mtf_confluence_row_safe"] = {
        "feature_name": "mtf_confluence_row_safe",
        "category": "diagnostic",
        "source_module": "p3k",
        "live_safe": False,
        "ml_safe": False,
        "onnx_safe": False,
        "forbidden": False,
        "base_timeframes": list(MTF_JOBS.keys()),
        "data_type": "int8",
        "lookahead_risk": "none",
        "filter_role": "required_filter_for_mtf_confluence_features",
        "reason": "Filter/diagnostic column. Use to gate P3K confluence features; do not feed into model X.",
    }

    features["mtf_confluence_safety_reason"] = {
        "feature_name": "mtf_confluence_safety_reason",
        "category": "diagnostic",
        "source_module": "p3k",
        "live_safe": False,
        "ml_safe": False,
        "onnx_safe": False,
        "forbidden": False,
        "base_timeframes": list(MTF_JOBS.keys()),
        "data_type": "string",
        "lookahead_risk": "none",
        "filter_role": "audit_reason_for_mtf_confluence_row_safe",
        "reason": "Diagnostic reason for P3K confluence safety. Not a model feature.",
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
        "blocked_features": sorted(blocked_features),
        "forbidden_live_features": sorted(FORBIDDEN_LIVE_FEATURES),
    }

    allowlists = {
        "ml": {
            **base,
            "purpose": "Numeric derived columns allowed for ML training input X after base, structure, MTF context, and confluence filters.",
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
        "mtf_structure_alignment_score": [1, -1],
        "mtf_swing_high_confluence_count": [0, 2],
        "mtf_structure_signal_score": [1.25, -0.75],
    })

    bad_df = pd.DataFrame({
        "mtf_structure_alignment_score": [1, -1],
        "mtf_confluence_row_safe": [1, 1],
        "mtf_confluence_safety_reason": ["SAFE", "SAFE"],
        "bars_to_next_large_gap": [5.0, 6.0],
    })

    rows = []

    try:
        assert_no_forbidden_features(good_df, registry=registry, mode="ml")
        rows.append({"test": "p3k_confluence_model_features_allowed", "status": "PASS", "message": "P3K model features allowed."})
    except Exception as e:
        rows.append({"test": "p3k_confluence_model_features_allowed", "status": "FAIL", "message": str(e)})

    try:
        assert_no_forbidden_features(bad_df, registry=registry, mode="ml")
        rows.append({"test": "p3k_diagnostics_blocked_from_ml", "status": "FAIL", "message": "Expected leakage error but none was raised."})
    except FeatureLeakageError as e:
        rows.append({"test": "p3k_diagnostics_blocked_from_ml", "status": "PASS", "message": str(e)})
    except Exception as e:
        rows.append({"test": "p3k_diagnostics_blocked_from_ml", "status": "FAIL", "message": str(e)})

    return pd.DataFrame(rows)


def write_report(
    summary_df: pd.DataFrame,
    registry: dict[str, Any],
    allowlists: dict[str, dict[str, Any]],
    guard_tests: pd.DataFrame,
) -> None:
    registry_rows = []
    for col in P3K_FEATURES:
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
            "filter_role": meta.get("filter_role"),
            "filter_required_for_use": meta.get("filter_required_for_use"),
        })

    pd.DataFrame(registry_rows).to_csv(REGISTRY_CHECK_CSV, index=False)
    guard_tests.to_csv(GUARD_TEST_CSV, index=False)

    lines = []
    lines.append("# P3K MTF Swing Confluence Zones Report")
    lines.append("")
    lines.append(f"Created UTC: `{now_utc()}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Added MTF support/resistance confluence features.")
    lines.append("- Added MTF structure vote features.")
    lines.append("- Added MTF BOS/CHoCH vote features.")
    lines.append("- Added `mtf_confluence_row_safe` and `mtf_confluence_safety_reason`.")
    lines.append("- Registered model-safe P3K features as `mtf_confluence`.")
    lines.append("- Kept confluence safety columns diagnostic/filter-only.")
    lines.append("- Rebuilt ML/live/ONNX allowlists.")
    lines.append("")
    lines.append("## Required filters")
    lines.append("")
    lines.append("```python")
    lines.append("df = df[df['is_feature_row_safe'] == 1]")
    lines.append("df = df[df['confirmed_structure_row_safe'] == 1]")
    lines.append("df = df[df['mtf_context_row_safe'] == 1]")
    lines.append("df = df[df['mtf_confluence_row_safe'] == 1]")
    lines.append("```")
    lines.append("")
    lines.append("## Dataset summary")
    lines.append("")
    lines.append("| Base TF | Context TFs | Rows | Cols Before | Cols After | Safe Rows | Safe Ratio | Near Resistance | Near Support | Bullish Vote Rows | Bearish Vote Rows | Conflict Rows | Inf Cells |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['base_timeframe']} | {row['context_timeframes']} | {int(row['rows'])} | "
            f"{int(row['cols_before'])} | {int(row['cols_after'])} | "
            f"{int(row['mtf_confluence_safe_rows'])} | {float(row['mtf_confluence_safe_ratio']):.6f} | "
            f"{int(row['near_resistance_rows'])} | {int(row['near_support_rows'])} | "
            f"{int(row['bullish_vote_rows'])} | {int(row['bearish_vote_rows'])} | "
            f"{int(row['conflict_rows'])} | {int(row['numeric_inf_cells'])} |"
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

    print("Adding P3K MTF swing confluence zones...")

    for base_tf, context_tfs in MTF_JOBS.items():
        path = mtf_path(base_tf)
        if not path.exists():
            raise FileNotFoundError(f"Missing P3J/P3J2 MTF dataset: {path}")

        df = pd.read_parquet(path)
        df, summary = compute_confluence(df, base_tf, context_tfs)
        df.to_parquet(path, index=False)

        summaries.append(summary)
        print(
            f"{base_tf}: safe={summary['mtf_confluence_safe_rows']}/{summary['rows']} "
            f"ratio={summary['mtf_confluence_safe_ratio']:.4f} cols={summary['cols_before']}->{summary['cols_after']}"
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
        "phase": "P3K",
        "created_at_utc": now_utc(),
        "datasets_patched": len(summary_df),
        "p3k_model_features_added": len(MODEL_SAFE_P3K_FEATURES),
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

    print("P3K COMPLETE")
    print(json.dumps(event, indent=2))


if __name__ == "__main__":
    main()
