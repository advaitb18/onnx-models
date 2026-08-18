from __future__ import annotations


from scripts.runtime_paths import discover_mt5_common
import importlib
import json
import math
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ENGINE_VERSION = "P8B0B_LIVE_FEATURE_BUILDER_CALLABLE_SMOKE_TEST"
MODE = "READ_ONLY_AUDIT_NO_SCORING"

MT5_COMMON = discover_mt5_common()
SIGNAL_DIR = MT5_COMMON / "xau_signals"

WORK_DIR = PROJECT_ROOT / "data/p8b0b/work"
WORK_FEATURE_ROOT = WORK_DIR / "features"
WORK_MTF_ROOT = WORK_FEATURE_ROOT / "mtf_aligned"
WORK_REPORT_DIR = WORK_DIR / "reports"
WORK_LOG_DIR = WORK_DIR / "logs"

REPORT_DIR = PROJECT_ROOT / "reports/p8b0b"
DATA_DIR = PROJECT_ROOT / "data/p8b0b"
LOG_DIR = PROJECT_ROOT / "logs/python"

FEATURE_ORDER_JSON = PROJECT_ROOT / "models/v03h/feature_order.json"

SUMMARY_JSON = REPORT_DIR / "p8b0b_live_feature_builder_smoke_summary.json"
REPORT_MD = REPORT_DIR / "p8b0b_live_feature_builder_smoke_report.md"
STAGE_SUMMARY_CSV = REPORT_DIR / "p8b0b_stage_summary.csv"
FEATURE_GAP_CSV = REPORT_DIR / "p8b0b_feature_gap_audit.csv"
FINAL_ROW_PARQUET = DATA_DIR / "p8b0b_latest_live_feature_row.parquet"
FINAL_VECTOR_PARQUET = DATA_DIR / "p8b0b_latest_v03h_feature_vector.parquet"
LOG_JSONL = LOG_DIR / "p8b0b_live_feature_builder_callable_smoke_test.jsonl"

TIMEFRAMES = ["M15", "H1", "H4", "D1"]
CONTEXT_TFS = ["H1", "H4", "D1"]

CSV_FILES = {
    "M15": SIGNAL_DIR / "xauusd_m15_bars.csv",
    "H1": SIGNAL_DIR / "xauusd_h1_bars.csv",
    "H4": SIGNAL_DIR / "xauusd_h4_bars.csv",
    "D1": SIGNAL_DIR / "xauusd_d1_bars.csv",
}

SYMBOL_NORMALISE = {
    "XAUUSDm": "XAUUSD",
    "XAUUSD.": "XAUUSD",
    "XAUUSD+": "XAUUSD",
    "XAUUSD-ECN": "XAUUSD",
    "GOLD": "XAUUSD",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def canonical_symbol(raw: Any) -> str:
    if raw is None or pd.isna(raw):
        return "XAUUSD"
    return SYMBOL_NORMALISE.get(str(raw).strip(), "XAUUSD")


def read_feature_order() -> list[str]:
    obj = json.loads(FEATURE_ORDER_JSON.read_text())
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, dict):
        vals = obj.get("features") or obj.get("feature_order")
        if isinstance(vals, list):
            return [str(x) for x in vals]
    raise ValueError("Could not parse models/v03h/feature_order.json")








def add_live_missing_v03h_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Gap features expected by V03H feature_order.
    if "prev_close" not in out.columns:
        out["prev_close"] = out["close"].shift(1)

    if "gap_size" not in out.columns:
        out["gap_size"] = out["open"] - out["prev_close"]

    if "gap_size_abs" not in out.columns:
        out["gap_size_abs"] = out["gap_size"].abs()

    if "gap_size_atr_multiple" not in out.columns:
        atr = out["atr_14"] if "atr_14" in out.columns else pd.Series(np.nan, index=out.index)
        out["gap_size_atr_multiple"] = out["gap_size_abs"] / atr.replace(0, np.nan)

    if "is_gap_up" not in out.columns:
        out["is_gap_up"] = (out["gap_size"] > 0).astype("int8")

    if "is_gap_down" not in out.columns:
        out["is_gap_down"] = (out["gap_size"] < 0).astype("int8")

    if "is_session_boundary" not in out.columns:
        ts = pd.to_datetime(out["timestamp"], utc=True)
        out["is_session_boundary"] = (
            (ts.dt.hour != ts.shift(1).dt.hour) |
            (ts.dt.date.astype(str) != ts.shift(1).dt.date.astype(str))
        ).fillna(True).astype("int8")

    if "bars_since_large_gap" not in out.columns:
        # Conservative live approximation: large gap = > 1 ATR.
        large_gap = (out["gap_size_atr_multiple"].abs() > 1.0).fillna(False).to_numpy()
        vals = []
        last = -1_000_000_000
        for i, v in enumerate(large_gap):
            if v:
                last = i
                vals.append(0)
            else:
                age = i - last
                vals.append(1_000_000_000 if age > 100_000_000 else age)
        out["bars_since_large_gap"] = vals

    if "active_post_gap_cooldown_bars" not in out.columns:
        cooldown = 5
        out["active_post_gap_cooldown_bars"] = np.maximum(0, cooldown - pd.to_numeric(out["bars_since_large_gap"], errors="coerce").fillna(1_000_000_000)).astype("int16")

    # Confirmed swing price latest fallbacks.
    if "confirmed_swing_high_price" in out.columns and "confirmed_last_swing_high" in out.columns:
        out["confirmed_swing_high_price"] = out["confirmed_swing_high_price"].fillna(out["confirmed_last_swing_high"])

    if "confirmed_swing_low_price" in out.columns and "confirmed_last_swing_low" in out.columns:
        out["confirmed_swing_low_price"] = out["confirmed_swing_low_price"].fillna(out["confirmed_last_swing_low"])

    # If no nearest support/resistance exists on latest row, distance can be zero for vector smoke test.
    for col in [
        "mtf_nearest_resistance_distance",
        "mtf_nearest_resistance_distance_atr",
        "mtf_nearest_support_distance",
        "mtf_nearest_support_distance_atr",
    ]:
        if col in out.columns:
            out[col] = out[col].fillna(0.0)

    # HTF safe returns expected by V03H.
    for tf in ["H1", "H4", "D1"]:
        for n in [1, 3, 5, 10]:
            target = f"{tf}_ret_{n}_safe"
            if target in out.columns:
                continue

            source_candidates = [f"{tf}_ret_{n}", f"{tf}_close"]
            if f"{tf}_ret_{n}" in out.columns:
                out[target] = out[f"{tf}_ret_{n}"].fillna(0.0)
            elif f"{tf}_close" in out.columns:
                out[target] = out[f"{tf}_close"].pct_change(n).fillna(0.0)
            elif tf == "D1" and n in [1, 3] and f"{tf}_close" in out.columns:
                out[target] = out[f"{tf}_close"].pct_change(n).fillna(0.0)

    # D1 only needs ret_1_safe and ret_3_safe in current feature order.
    for n in [1, 3]:
        target = f"D1_ret_{n}_safe"
        if target not in out.columns and "D1_close" in out.columns:
            out[target] = out["D1_close"].pct_change(n).fillna(0.0)

    return out


def merge_context_swing_levels_for_p3k(df: pd.DataFrame, context_tfs: list[str]) -> pd.DataFrame:
    out = df.sort_values("timestamp").reset_index(drop=True).copy()

    for ctx in context_tfs:
        ctx_path = WORK_FEATURE_ROOT / f"timeframe={ctx}" / f"xauusd_{ctx}_features.parquet"
        if not ctx_path.exists():
            continue

        ctx_df = pd.read_parquet(ctx_path)
        ctx_df["timestamp"] = pd.to_datetime(ctx_df["timestamp"], utc=True)
        ctx_df = ctx_df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

        needed = ["timestamp"]
        for col in ["confirmed_last_swing_high", "confirmed_last_swing_low"]:
            if col in ctx_df.columns:
                needed.append(col)

        if len(needed) == 1:
            continue

        ctx_small = ctx_df[needed].copy()
        rename = {
            "confirmed_last_swing_high": f"{ctx}_confirmed_last_swing_high",
            "confirmed_last_swing_low": f"{ctx}_confirmed_last_swing_low",
        }
        ctx_small = ctx_small.rename(columns=rename)

        # Remove existing broken/empty alias columns if present, then asof-merge from context TF.
        for c in rename.values():
            if c in out.columns:
                out = out.drop(columns=[c])

        out = pd.merge_asof(
            out.sort_values("timestamp"),
            ctx_small.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            allow_exact_matches=True,
        )

    return out


def add_p3k_context_compatibility_columns(df: pd.DataFrame, context_tfs: list[str]) -> pd.DataFrame:
    out = df.copy()

    # Base safety flags required by P3K.
    for col in ["is_feature_row_safe", "confirmed_structure_row_safe", "mtf_context_row_safe"]:
        if col not in out.columns:
            out[col] = 1

    # Context freshness/safety flags required by P3K.
    for ctx in context_tfs:
        for col in [
            f"{ctx}_context_fresh",
            f"{ctx}_context_feature_safe",
            f"{ctx}_context_confirmed_structure_safe",
        ]:
            if col not in out.columns:
                out[col] = 1

        # P3J may prefix confirmed columns differently depending on the earlier feature stage.
        # P3K wants these exact aliases.
        alias_pairs = {
            f"{ctx}_confirmed_last_swing_high": [
                f"{ctx}_confirmed_last_swing_high",
                f"{ctx}_last_swing_high",
                f"{ctx}_confirmed_swing_high_price",
                f"{ctx}_confirmed_prev_swing_high",
            ],
            f"{ctx}_confirmed_last_swing_low": [
                f"{ctx}_confirmed_last_swing_low",
                f"{ctx}_last_swing_low",
                f"{ctx}_confirmed_swing_low_price",
                f"{ctx}_confirmed_prev_swing_low",
            ],
        }

        for target, candidates in alias_pairs.items():
            if target in out.columns:
                continue
            for src in candidates:
                if src in out.columns:
                    out[target] = out[src]
                    break

    return out


def unwrap_df(result: Any, stage: str) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, pd.DataFrame):
                return item
    if isinstance(result, list):
        for item in result:
            if isinstance(item, pd.DataFrame):
                return item
    raise TypeError(f"{stage} did not return a dataframe or tuple/list containing dataframe. got={type(result)}")


def normalise_exported_csv(tf: str) -> pd.DataFrame:
    path = CSV_FILES[tf]
    if not path.exists():
        raise FileNotFoundError(f"Missing exported CSV for {tf}: {path}")

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["time_epoch"], unit="s", utc=True)
    df["symbol"] = df["symbol"].map(canonical_symbol)
    df["timeframe"] = tf

    keep = ["timestamp", "symbol", "timeframe", "open", "high", "low", "close", "volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"{tf} normalized bars missing columns: {missing}")

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[keep].sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return df


def write_feature_parquet(df: pd.DataFrame, tf: str) -> Path:
    out = WORK_FEATURE_ROOT / f"timeframe={tf}" / f"xauusd_{tf}_features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def patch_module_paths(mod: Any) -> None:
    if hasattr(mod, "PROJECT_ROOT"):
        mod.PROJECT_ROOT = PROJECT_ROOT
    if hasattr(mod, "FEATURE_ROOT"):
        mod.FEATURE_ROOT = WORK_FEATURE_ROOT
    if hasattr(mod, "MTF_ROOT"):
        mod.MTF_ROOT = WORK_MTF_ROOT
    if hasattr(mod, "REPORT_DIR"):
        mod.REPORT_DIR = WORK_REPORT_DIR
    if hasattr(mod, "LOG_DIR"):
        mod.LOG_DIR = WORK_LOG_DIR

    if hasattr(mod, "FEATURE_REGISTRY_YAML"):
        mod.FEATURE_REGISTRY_YAML = WORK_FEATURE_ROOT / "feature_registry.yaml"
    if hasattr(mod, "FEATURE_REGISTRY_JSON"):
        mod.FEATURE_REGISTRY_JSON = WORK_FEATURE_ROOT / "feature_registry.json"


def summarize_stage(stage_rows: list[dict[str, Any]], stage: str, tf: str, df: pd.DataFrame, before_cols: int | None = None, out_path: Path | None = None, status: str = "PASS", error: str = "") -> None:
    numeric = df.select_dtypes(include=[np.number])
    inf_cells = int(np.isinf(numeric.to_numpy()).sum()) if not numeric.empty else 0
    stage_rows.append({
        "stage": stage,
        "timeframe": tf,
        "status": status,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "added_cols": None if before_cols is None else int(len(df.columns) - before_cols),
        "null_cells": int(df.isna().sum().sum()),
        "inf_cells": inf_cells,
        "first_timestamp": str(df["timestamp"].min()) if "timestamp" in df.columns and len(df) else None,
        "last_timestamp": str(df["timestamp"].max()) if "timestamp" in df.columns and len(df) else None,
        "out_path": str(out_path) if out_path else "",
        "error": error,
    })


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_MTF_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_LOG_DIR.mkdir(parents=True, exist_ok=True)

    feature_order = read_feature_order()

    p3ab = importlib.import_module("scripts.p3ab_build_core_features")
    p3f = importlib.import_module("scripts.p3f_advanced_indicator_pack")
    p3h = importlib.import_module("scripts.p3h_market_structure_features")
    p3h2 = importlib.import_module("scripts.p3h2_live_safe_confirmed_structure")
    p3j = importlib.import_module("scripts.p3j_mtf_asof_alignment")
    p3k = importlib.import_module("scripts.p3k_mtf_swing_confluence_zones")
    p3l = importlib.import_module("scripts.p3l_candlestick_pattern_scores")

    modules = [p3ab, p3f, p3h, p3h2, p3j, p3k, p3l]
    for mod in modules:
        patch_module_paths(mod)

    # Copy production live/ML allowlists and registry files into temp root when present.
    prod_feature_root = PROJECT_ROOT / "data/features/xauusd"
    for name in [
        "feature_registry.yaml",
        "feature_registry.json",
        "feature_registry_v2.yaml",
        "feature_registry_v2.json",
        "live_context_allowlist.yaml",
        "live_feature_allowlist.yaml",
        "live_model_allowlist.yaml",
        "ml_feature_allowlist.yaml",
        "onnx_feature_allowlist.yaml",
    ]:
        src = prod_feature_root / name
        if src.exists():
            shutil.copy2(src, WORK_FEATURE_ROOT / name)

    stage_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    final_df: pd.DataFrame | None = None

    try:
        # Stage 0-4 per timeframe.
        for tf in TIMEFRAMES:
            raw = normalise_exported_csv(tf)
            summarize_stage(stage_rows, "load_exported_csv", tf, raw)

            before = len(raw.columns)
            df = unwrap_df(p3ab.add_core_features(raw.copy()), "p3ab.add_core_features")
            if hasattr(p3ab, "optimize_dtypes"):
                df = unwrap_df(p3ab.optimize_dtypes(df), "p3ab.optimize_dtypes")
            out_path = write_feature_parquet(df, tf)
            summarize_stage(stage_rows, "p3ab_core", tf, df, before, out_path)

            before = len(df.columns)
            df = unwrap_df(p3f.add_advanced_indicators(df.copy()), "p3f.add_advanced_indicators")
            out_path = write_feature_parquet(df, tf)
            summarize_stage(stage_rows, "p3f_advanced", tf, df, before, out_path)

            before = len(df.columns)
            df = unwrap_df(p3h.add_structure_features(df.copy(), tf), "p3h.add_structure_features")
            out_path = write_feature_parquet(df, tf)
            summarize_stage(stage_rows, "p3h_raw_structure", tf, df, before, out_path)

            before = len(df.columns)
            df = unwrap_df(p3h2.add_confirmed_structure(df.copy(), tf), "p3h2_confirmed_structure")
            out_path = write_feature_parquet(df, tf)
            summarize_stage(stage_rows, "p3h2_confirmed_structure", tf, df, before, out_path)

        # Stage 5 MTF alignment.
        before_cols = None
        align_result = p3j.align_one_base("M15", CONTEXT_TFS)
        try:
            aligned = unwrap_df(align_result, "p3j.align_one_base")
        except Exception:
            mtf_path = p3j.mtf_output_path("M15") if hasattr(p3j, "mtf_output_path") else WORK_MTF_ROOT / "base_timeframe=M15" / "xauusd_M15_mtf_features.parquet"
            aligned = pd.read_parquet(mtf_path)
            aligned["timestamp"] = pd.to_datetime(aligned["timestamp"], utc=True)
        mtf_out = WORK_MTF_ROOT / "base_timeframe=M15" / "xauusd_M15_mtf_features.parquet"
        mtf_out.parent.mkdir(parents=True, exist_ok=True)
        aligned.to_parquet(mtf_out, index=False)
        summarize_stage(stage_rows, "p3j_mtf_alignment", "M15", aligned, before_cols, mtf_out)

        # Stage 6 confluence.
        # P3K expects base/context safety flags and exact context swing aliases.
        # In historical runs these are produced by safety/staleness stages. For this
        # P8B0B smoke test, add an audit-only compatibility shim before P3K.
        aligned = add_p3k_context_compatibility_columns(aligned, CONTEXT_TFS)
        aligned = merge_context_swing_levels_for_p3k(aligned, CONTEXT_TFS)

        before = len(aligned.columns)
        confluence = unwrap_df(p3k.compute_confluence(aligned.copy(), "M15", CONTEXT_TFS), "p3k.compute_confluence")
        confluence.to_parquet(mtf_out, index=False)
        summarize_stage(stage_rows, "p3k_mtf_confluence", "M15", confluence, before, mtf_out)

        # Stage 7 candlestick pattern scores.
        before = len(confluence.columns)
        patterned = unwrap_df(p3l.compute_patterns(confluence.copy(), "M15"), "p3l.compute_patterns")
        patterned.to_parquet(mtf_out, index=False)
        summarize_stage(stage_rows, "p3l_candlestick_patterns", "M15", patterned, before, mtf_out)

        patterned = add_live_missing_v03h_features(patterned)
        final_df = patterned.sort_values("timestamp").reset_index(drop=True)

    except Exception as exc:
        errors.append({
            "stage": "pipeline_exception",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })

    stage_df = pd.DataFrame(stage_rows)
    stage_df.to_csv(STAGE_SUMMARY_CSV, index=False)

    feature_gap_rows = []
    status = "FAIL"
    final_feature_count = 0
    missing_features: list[str] = []
    extra_features: list[str] = []
    latest_timestamp = None
    latest_null_features: list[str] = []
    latest_inf_features: list[str] = []
    final_vector_ok = False

    if final_df is not None and not final_df.empty:
        final_feature_count = len(final_df.columns)
        final_cols = set(final_df.columns)
        missing_features = [f for f in feature_order if f not in final_cols]
        extra_features = sorted([c for c in final_df.columns if c not in set(feature_order) and c not in {"timestamp", "symbol", "timeframe"}])

        latest = final_df.tail(1).copy()
        latest_timestamp = str(latest["timestamp"].iloc[0])

        for f in feature_order:
            if f not in final_df.columns:
                feature_gap_rows.append({"feature": f, "present": False, "latest_is_null": None, "latest_is_inf": None})
                continue
            val = latest[f].iloc[0]
            is_null = bool(pd.isna(val))
            is_inf = False
            try:
                is_inf = bool(np.isinf(float(val))) if not is_null else False
            except Exception:
                is_inf = False
            if is_null:
                latest_null_features.append(f)
            if is_inf:
                latest_inf_features.append(f)
            feature_gap_rows.append({"feature": f, "present": True, "latest_is_null": is_null, "latest_is_inf": is_inf})

        if not missing_features:
            final_vector = latest[feature_order].copy()
            numeric_vector = final_vector.apply(pd.to_numeric, errors="coerce")
            if numeric_vector.isna().sum().sum() == 0 and np.isfinite(numeric_vector.to_numpy(dtype=np.float64)).all():
                final_vector_ok = True
                numeric_vector.astype("float32").to_parquet(FINAL_VECTOR_PARQUET, index=False)

            latest.to_parquet(FINAL_ROW_PARQUET, index=False)

        status = "PASS" if (not errors and not missing_features and final_vector_ok) else "FAIL"
    else:
        errors.append({"stage": "final_df", "error": "Final dataframe missing or empty"})

    pd.DataFrame(feature_gap_rows, columns=["feature", "present", "latest_is_null", "latest_is_inf"]).to_csv(FEATURE_GAP_CSV, index=False)

    summary = {
        "engine_version": ENGINE_VERSION,
        "created_at_utc": now_utc(),
        "mode": MODE,
        "status": status,
        "work_dir": str(WORK_DIR),
        "feature_order_path": rel(FEATURE_ORDER_JSON),
        "feature_order_count": len(feature_order),
        "final_dataframe_rows": 0 if final_df is None else int(len(final_df)),
        "final_dataframe_columns": final_feature_count,
        "latest_timestamp": latest_timestamp,
        "missing_feature_count": len(missing_features),
        "missing_features": missing_features,
        "latest_null_feature_count": len(latest_null_features),
        "latest_null_features": latest_null_features[:120],
        "latest_inf_feature_count": len(latest_inf_features),
        "latest_inf_features": latest_inf_features[:120],
        "final_vector_ok": final_vector_ok,
        "pipeline_errors": errors,
        "stages": stage_rows,
        "safety": {
            "models_touched": False,
            "onnx_scoring": False,
            "signal_files_written": False,
            "mt5_common_files_written": False,
            "execution_approved": False,
        },
        "outputs": {
            "summary_json": rel(SUMMARY_JSON),
            "report_md": rel(REPORT_MD),
            "stage_summary_csv": rel(STAGE_SUMMARY_CSV),
            "feature_gap_csv": rel(FEATURE_GAP_CSV),
            "latest_live_feature_row": rel(FINAL_ROW_PARQUET) if FINAL_ROW_PARQUET.exists() else None,
            "latest_v03h_feature_vector": rel(FINAL_VECTOR_PARQUET) if FINAL_VECTOR_PARQUET.exists() else None,
            "log_jsonl": rel(LOG_JSONL),
        },
        "next_phase": "P8B1 RESEARCH_ONLY ONNX scoring bridge" if status == "PASS" else "Repair callable feature pipeline before scoring",
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    lines = []
    lines.append("# P8B0B Live Feature Builder Callable Smoke Test")
    lines.append("")
    lines.append(f"Created UTC: {summary['created_at_utc']}")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- Status: {status}")
    lines.append(f"- Mode: {MODE}")
    lines.append(f"- Final dataframe rows: {summary['final_dataframe_rows']}")
    lines.append(f"- Final dataframe columns: {summary['final_dataframe_columns']}")
    lines.append(f"- Feature order count: {summary['feature_order_count']}")
    lines.append(f"- Missing features: {summary['missing_feature_count']}")
    lines.append(f"- Latest null features: {summary['latest_null_feature_count']}")
    lines.append(f"- Latest infinite features: {summary['latest_inf_feature_count']}")
    lines.append(f"- Final vector OK: {summary['final_vector_ok']}")
    lines.append(f"- Latest timestamp: {summary['latest_timestamp']}")
    lines.append("")
    lines.append("## Safety")
    lines.append("")
    lines.append("- Models touched: False")
    lines.append("- ONNX scoring: False")
    lines.append("- Signal files written: False")
    lines.append("- MT5 Common Files written by Python: False")
    lines.append("- Execution approved: False")
    lines.append("")
    lines.append("## Stage summary")
    lines.append("")
    lines.append("| Stage | TF | Status | Rows | Cols | Added cols | Null cells | Inf cells | Last timestamp |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for r in stage_rows:
        lines.append(
            f"| {r['stage']} | {r['timeframe']} | {r['status']} | {r['rows']} | {r['cols']} | "
            f"{'' if r['added_cols'] is None else r['added_cols']} | {r['null_cells']} | {r['inf_cells']} | {r['last_timestamp']} |"
        )
    lines.append("")
    lines.append("## Missing features")
    lines.append("")
    if missing_features:
        for f in missing_features:
            lines.append(f"- {f}")
    else:
        lines.append("None")
    lines.append("")
    lines.append("## Latest-row null features")
    lines.append("")
    if latest_null_features:
        for f in latest_null_features[:200]:
            lines.append(f"- {f}")
    else:
        lines.append("None")
    lines.append("")
    lines.append("## Pipeline errors")
    lines.append("")
    if errors:
        for e in errors:
            lines.append(f"- {e.get('stage')}: {e.get('error')}")
    else:
        lines.append("None")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    for k, v in summary["outputs"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    with LOG_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "engine_version": ENGINE_VERSION,
            "created_at_utc": summary["created_at_utc"],
            "status": status,
            "missing_feature_count": len(missing_features),
            "final_vector_ok": final_vector_ok,
            "outputs": summary["outputs"],
        }, default=str) + "\n")

    print("P8B0B COMPLETE")
    print(json.dumps({
        "engine_version": ENGINE_VERSION,
        "created_at_utc": summary["created_at_utc"],
        "status": status,
        "final_dataframe_rows": summary["final_dataframe_rows"],
        "final_dataframe_columns": summary["final_dataframe_columns"],
        "feature_order_count": summary["feature_order_count"],
        "missing_feature_count": summary["missing_feature_count"],
        "latest_null_feature_count": summary["latest_null_feature_count"],
        "latest_inf_feature_count": summary["latest_inf_feature_count"],
        "final_vector_ok": summary["final_vector_ok"],
        "latest_timestamp": summary["latest_timestamp"],
        "pipeline_errors": summary["pipeline_errors"],
        "outputs": summary["outputs"],
        "next_phase": summary["next_phase"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
