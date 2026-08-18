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
LIVE_CONTEXT_ALLOWLIST_YAML = FEATURE_ROOT / "live_context_allowlist.yaml"
LIVE_MODEL_ALLOWLIST_YAML = FEATURE_ROOT / "live_model_allowlist.yaml"
ONNX_ALLOWLIST_YAML = FEATURE_ROOT / "onnx_feature_allowlist.yaml"

REPORT_MD = REPORT_DIR / "p3j_mtf_asof_alignment_report.md"
SUMMARY_CSV = REPORT_DIR / "p3j_mtf_asof_alignment_summary.csv"
REGISTRY_CHECK_CSV = REPORT_DIR / "p3j_mtf_registry_columns.csv"
GUARD_TEST_CSV = REPORT_DIR / "p3j_leakage_guard_smoke_test.csv"
LOG_JSONL = LOG_DIR / "p3j_mtf_asof_alignment.jsonl"

TIMEFRAME_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
}

MTF_JOBS = {
    # lower timeframe dataset -> higher timeframe context features
    "M15": ["H1", "H4", "D1"],
    "M5": ["M15", "H1", "H4"],
    "M1": ["M5", "M15", "H1"],
}

# Conservative higher-timeframe feature set for MTF context.
# These are already live-safe and model-safe after P3I/P3H2.
MTF_CONTEXT_BASE_COLS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "is_feature_row_safe",
    "confirmed_structure_row_safe",

    "ret_1_safe",
    "ret_3_safe",
    "ret_5_safe",
    "ret_10_safe",

    "atr_14",
    "atr_pct_of_close",
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "adx_14",
    "plus_di_14",
    "minus_di_14",
    "bb_width_20",
    "bb_percent_b",
    "stoch_k",
    "stoch_d",

    "volatility_regime_low",
    "volatility_regime_mid",
    "volatility_regime_high",
    "trend_regime_bull",
    "trend_regime_bear",
    "trend_regime_sideways",

    "confirmed_bos_bull",
    "confirmed_bos_bear",
    "confirmed_choch_bull",
    "confirmed_choch_bear",
    "confirmed_structure_score_net",
    "confirmed_structure_trend_code",
    "confirmed_structure_age_bars",
    "confirmed_distance_to_last_swing_high_atr",
    "confirmed_distance_to_last_swing_low_atr",
    "confirmed_near_last_swing_high",
    "confirmed_near_last_swing_low",
    "confirmed_structure_momentum_3",
    "confirmed_structure_momentum_5",
    "confirmed_structure_momentum_10",
]

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


def mtf_output_path(base_tf: str) -> Path:
    return MTF_ROOT / f"base_timeframe={base_tf}" / f"xauusd_{base_tf}_mtf_features.parquet"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def load_features(tf: str) -> pd.DataFrame:
    path = feature_path(tf)
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{tf}: missing timestamp")
    return df.sort_values("timestamp").reset_index(drop=True)


def get_existing_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def add_available_at_timestamp(htf: pd.DataFrame, tf: str) -> pd.DataFrame:
    out = htf.copy()
    seconds = TIMEFRAME_SECONDS[tf]

    # Critical anti-leak rule:
    # A higher-timeframe candle is only available after its own close.
    # If H1 timestamp is candle open, H1 available_at = timestamp + 1 hour.
    out["available_at_timestamp"] = out["timestamp"] + pd.to_timedelta(seconds, unit="s")
    return out


def prefix_htf_columns(htf: pd.DataFrame, htf_name: str) -> pd.DataFrame:
    keep_cols = get_existing_cols(htf, MTF_CONTEXT_BASE_COLS + ["available_at_timestamp"])
    out = htf[keep_cols].copy()

    rename = {}
    for col in out.columns:
        if col == "available_at_timestamp":
            rename[col] = f"{htf_name}_available_at_timestamp"
        elif col == "timestamp":
            rename[col] = f"{htf_name}_source_timestamp"
        else:
            rename[col] = f"{htf_name}_{col}"

    return out.rename(columns=rename)


def align_one_base(base_tf: str, context_tfs: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = load_features(base_tf)

    # Lower timeframe row timestamp is the decision-time anchor.
    # We do not mutate base columns.
    aligned = base.sort_values("timestamp").reset_index(drop=True)

    base_rows = len(aligned)
    base_cols_before = len(aligned.columns)

    context_summaries = []

    for htf_name in context_tfs:
        htf = load_features(htf_name)
        htf = add_available_at_timestamp(htf, htf_name)
        htf_prefixed = prefix_htf_columns(htf, htf_name).sort_values(f"{htf_name}_available_at_timestamp")

        before_cols = len(aligned.columns)

        aligned = pd.merge_asof(
            aligned.sort_values("timestamp"),
            htf_prefixed,
            left_on="timestamp",
            right_on=f"{htf_name}_available_at_timestamp",
            direction="backward",
            allow_exact_matches=True,
        )

        after_cols = len(aligned.columns)

        available_col = f"{htf_name}_available_at_timestamp"
        source_col = f"{htf_name}_source_timestamp"

        matched_rows = int(aligned[available_col].notna().sum())
        match_ratio = float(matched_rows / base_rows) if base_rows else 0.0

        # Leakage check: available_at must never be after the base timestamp.
        leakage_rows = int((aligned[available_col] > aligned["timestamp"]).fillna(False).sum())

        # Also check source candle open is strictly before or equal its available timestamp.
        bad_source_rows = int((aligned[source_col] >= aligned["timestamp"]).fillna(False).sum())

        context_summaries.append({
            "base_timeframe": base_tf,
            "context_timeframe": htf_name,
            "matched_rows": matched_rows,
            "match_ratio": match_ratio,
            "cols_added": int(after_cols - before_cols),
            "available_at_after_base_timestamp_rows": leakage_rows,
            "source_timestamp_not_before_base_rows": bad_source_rows,
            "first_available_at": str(aligned[available_col].dropna().min()) if matched_rows else None,
            "last_available_at": str(aligned[available_col].dropna().max()) if matched_rows else None,
        })

    out_path = mtf_output_path(base_tf)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_parquet(out_path, index=False)

    summary = {
        "base_timeframe": base_tf,
        "context_timeframes": ",".join(context_tfs),
        "rows": int(len(aligned)),
        "base_cols_before": int(base_cols_before),
        "cols_after": int(len(aligned.columns)),
        "cols_added": int(len(aligned.columns) - base_cols_before),
        "output_path": rel(out_path),
        "context_summaries": context_summaries,
    }

    return aligned, summary


def collect_mtf_columns() -> dict[str, Any]:
    mtf_cols: dict[str, Any] = {}

    for base_tf in MTF_JOBS:
        path = mtf_output_path(base_tf)
        df = pd.read_parquet(path)

        for col in df.columns:
            if not any(col.startswith(f"{ctx}_") for ctx in MTF_JOBS[base_tf]):
                continue

            meta = mtf_cols.setdefault(
                col,
                {
                    "timeframes": [],
                    "data_types_by_base_timeframe": {},
                    "null_counts_by_base_timeframe": {},
                    "non_null_counts_by_base_timeframe": {},
                },
            )
            meta["timeframes"].append(base_tf)
            meta["data_types_by_base_timeframe"][base_tf] = str(df[col].dtype)
            meta["null_counts_by_base_timeframe"][base_tf] = int(df[col].isna().sum())
            meta["non_null_counts_by_base_timeframe"][base_tf] = int(df[col].notna().sum())

    return mtf_cols


def update_registry() -> dict[str, Any]:
    registry = load_yaml(REGISTRY_YAML)
    features = registry.setdefault("features", {})
    mtf_cols = collect_mtf_columns()

    registry["registry_revision"] = "P3J"
    registry["updated_at_utc"] = now_utc()
    registry.setdefault("policy", {})["mtf_alignment_asof_required"] = True
    registry.setdefault("policy", {})["mtf_higher_timeframe_available_at_required"] = True
    registry.setdefault("policy", {})["mtf_current_incomplete_higher_tf_forbidden"] = True
    registry["mtf_alignment_jobs"] = MTF_JOBS
    registry["mtf_timeframe_seconds"] = TIMEFRAME_SECONDS

    for col, meta in sorted(mtf_cols.items()):
        dtype_values = sorted(set(meta["data_types_by_base_timeframe"].values()))
        dtype = dtype_values[0] if len(dtype_values) == 1 else "mixed"

        # available_at/source timestamps are metadata/context, not model features.
        if col.endswith("_available_at_timestamp") or col.endswith("_source_timestamp"):
            category = "mtf_metadata"
            live_safe = False
            ml_safe = False
            onnx_safe = False
            reason = "MTF audit timestamp column. Used to verify as-of alignment; not model input."
        elif col.endswith("_symbol") or col.endswith("_timeframe") or col.endswith("_source") or col.endswith("_source_file"):
            category = "mtf_metadata"
            live_safe = False
            ml_safe = False
            onnx_safe = False
            reason = "MTF metadata column. Not model input."
        elif col.endswith("_is_feature_row_safe") or col.endswith("_confirmed_structure_row_safe"):
            category = "mtf_filter"
            live_safe = False
            ml_safe = False
            onnx_safe = False
            reason = "MTF higher-timeframe safety filter. Used for gating, not model input."
        else:
            category = "mtf"
            live_safe = True
            ml_safe = True
            onnx_safe = True
            reason = "As-of aligned higher-timeframe feature using only fully closed higher-timeframe candles."

        features[col] = {
            "feature_name": col,
            "category": category,
            "source_module": "p3j",
            "live_safe": live_safe,
            "ml_safe": ml_safe,
            "onnx_safe": onnx_safe,
            "forbidden": False,
            "base_timeframes": meta["timeframes"],
            "data_type": dtype,
            "data_types_by_base_timeframe": meta["data_types_by_base_timeframe"],
            "lookback_bars": 0,
            "requires_confirmation": False,
            "lookahead_risk": "none",
            "alignment_method": "merge_asof_backward",
            "available_at_rule": "higher_timeframe_timestamp_plus_timeframe_duration",
            "reason": reason,
            "null_counts_by_base_timeframe": meta["null_counts_by_base_timeframe"],
            "non_null_counts_by_base_timeframe": meta["non_null_counts_by_base_timeframe"],
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

    def is_numeric_model_candidate(meta: dict[str, Any]) -> bool:
        dtype = str(meta.get("data_type", ""))
        return not (
            dtype.startswith("object")
            or dtype.startswith("string")
            or dtype.startswith("datetime")
            or dtype.startswith("category")
        )

    ml_features = [
        name for name, meta in features.items()
        if not blocked(meta, "ml")
        and meta["category"] not in {"raw", "metadata", "diagnostic", "mtf_metadata", "mtf_filter"}
        and is_numeric_model_candidate(meta)
    ]

    live_context_features = [
        name for name, meta in features.items()
        if not blocked(meta, "live")
        and meta["category"] not in {"metadata", "diagnostic", "mtf_metadata", "mtf_filter"}
    ]

    live_model_features = [
        name for name, meta in features.items()
        if not blocked(meta, "live")
        and meta["category"] not in {"raw", "metadata", "diagnostic", "mtf_metadata", "mtf_filter"}
        and is_numeric_model_candidate(meta)
    ]

    onnx_features = [
        name for name, meta in features.items()
        if not blocked(meta, "onnx")
        and meta["category"] not in {"raw", "metadata", "diagnostic", "mtf_metadata", "mtf_filter"}
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
        "required_mtf_alignment_rule": "higher_tf_available_at_timestamp <= base_timestamp",
        "blocked_features": sorted(blocked_features),
        "forbidden_live_features": sorted(FORBIDDEN_LIVE_FEATURES),
    }

    allowlists = {
        "ml": {
            **base,
            "purpose": "Numeric derived columns allowed for ML training input X after filters.",
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

    # Keep legacy live allowlist in sync with live_context.
    write_yaml(FEATURE_ROOT / "live_feature_allowlist.yaml", allowlists["live_context"])

    return allowlists


def smoke_test_guard(registry: dict[str, Any]) -> pd.DataFrame:
    from xau_cgt.features.leakage_guard import FeatureLeakageError, assert_no_forbidden_features

    good_df = pd.DataFrame({
        "H1_rsi_14": [50.0, 55.0],
        "H4_confirmed_structure_score_net": [1.0, 2.0],
        "D1_atr_14": [20.0, 21.0],
    })

    bad_df = pd.DataFrame({
        "H1_rsi_14": [50.0, 55.0],
        "bars_to_next_large_gap": [1.0, 2.0],
        "H4_available_at_timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
    })

    rows = []

    try:
        assert_no_forbidden_features(good_df, registry=registry, mode="ml")
        rows.append({"test": "mtf_model_features_allowed", "status": "PASS", "message": "MTF model features allowed."})
    except Exception as e:
        rows.append({"test": "mtf_model_features_allowed", "status": "FAIL", "message": str(e)})

    try:
        assert_no_forbidden_features(bad_df, registry=registry, mode="ml")
        rows.append({"test": "future_gap_blocked", "status": "FAIL", "message": "Expected leakage error but none was raised."})
    except FeatureLeakageError as e:
        rows.append({"test": "future_gap_blocked", "status": "PASS", "message": str(e)})
    except Exception as e:
        rows.append({"test": "future_gap_blocked", "status": "FAIL", "message": str(e)})

    return pd.DataFrame(rows)


def write_report(
    summaries: list[dict[str, Any]],
    context_summary_df: pd.DataFrame,
    registry: dict[str, Any],
    allowlists: dict[str, dict[str, Any]],
    guard_tests: pd.DataFrame,
) -> None:
    mtf_registry_rows = []

    for name, meta in registry["features"].items():
        if meta.get("source_module") == "p3j":
            mtf_registry_rows.append({
                "feature_name": name,
                "category": meta.get("category"),
                "live_safe": meta.get("live_safe"),
                "ml_safe": meta.get("ml_safe"),
                "onnx_safe": meta.get("onnx_safe"),
                "lookahead_risk": meta.get("lookahead_risk"),
                "alignment_method": meta.get("alignment_method"),
                "available_at_rule": meta.get("available_at_rule"),
                "base_timeframes": ",".join(meta.get("base_timeframes", [])),
                "data_type": meta.get("data_type"),
            })

    pd.DataFrame(mtf_registry_rows).to_csv(REGISTRY_CHECK_CSV, index=False)
    guard_tests.to_csv(GUARD_TEST_CSV, index=False)

    lines = []
    lines.append("# P3J Multi-Timeframe As-Of Alignment Report")
    lines.append("")
    lines.append(f"Created UTC: `{now_utc()}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Created separate MTF-aligned feature parquet files.")
    lines.append("- Used strict as-of backward alignment.")
    lines.append("- Higher-timeframe features are only available after the higher-timeframe candle fully closes.")
    lines.append("- Added MTF registry entries with `alignment_method` and `available_at_rule`.")
    lines.append("- Rebuilt ML/live/ONNX allowlists.")
    lines.append("")
    lines.append("## Anti-leakage rule")
    lines.append("")
    lines.append("```text")
    lines.append("higher_tf_available_at_timestamp <= base_timestamp")
    lines.append("available_at_timestamp = higher_tf_timestamp + higher_tf_duration")
    lines.append("merge_asof direction = backward")
    lines.append("```")
    lines.append("")
    lines.append("## Dataset outputs")
    lines.append("")
    lines.append("| Base TF | Context TFs | Rows | Base Cols | Final Cols | Added Cols | Output |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for s in summaries:
        lines.append(
            f"| {s['base_timeframe']} | {s['context_timeframes']} | {s['rows']} | "
            f"{s['base_cols_before']} | {s['cols_after']} | {s['cols_added']} | `{s['output_path']}` |"
        )
    lines.append("")
    lines.append("## Context alignment checks")
    lines.append("")
    lines.append("| Base TF | Context TF | Match Ratio | Matched Rows | Available After Base Rows | Source Timestamp Not Before Base Rows |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for _, row in context_summary_df.iterrows():
        lines.append(
            f"| {row['base_timeframe']} | {row['context_timeframe']} | {float(row['match_ratio']):.6f} | "
            f"{int(row['matched_rows'])} | {int(row['available_at_after_base_timestamp_rows'])} | "
            f"{int(row['source_timestamp_not_before_base_rows'])} |"
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
    MTF_ROOT.mkdir(parents=True, exist_ok=True)

    summaries = []
    context_rows = []

    print("Building MTF as-of aligned datasets...")

    for base_tf, context_tfs in MTF_JOBS.items():
        _, summary = align_one_base(base_tf, context_tfs)
        summaries.append(summary)
        context_rows.extend(summary["context_summaries"])

        print(
            f"{base_tf}: rows={summary['rows']} cols "
            f"{summary['base_cols_before']}->{summary['cols_after']} contexts={summary['context_timeframes']}"
        )

    summary_df = pd.DataFrame([{k: v for k, v in s.items() if k != "context_summaries"} for s in summaries])
    context_summary_df = pd.DataFrame(context_rows)

    summary_df.to_csv(SUMMARY_CSV, index=False)

    print("Updating registry...")
    registry = update_registry()

    print("Rebuilding allowlists...")
    allowlists = rebuild_allowlists(registry)

    print("Running leakage guard tests...")
    guard_tests = smoke_test_guard(registry)

    print("Writing report...")
    write_report(summaries, context_summary_df, registry, allowlists, guard_tests)

    event = {
        "phase": "P3J",
        "created_at_utc": now_utc(),
        "datasets_built": len(summaries),
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

    print("P3J COMPLETE")
    print(json.dumps(event, indent=2))


if __name__ == "__main__":
    main()
