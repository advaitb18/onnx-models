#!/usr/bin/env python3
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

required = [
    "config/multi_strategy_portfolio.json",

    "models/v03h/final_rf_sell_model.onnx",
    "models/v03h/feature_order.json",
    "models/v03h/score_distribution.parquet",

    "models/v05h/final_rf_buy_model.onnx",
    "models/v05h/feature_order.json",
    "models/v05h/score_distribution.parquet",

    "mt5_ea/XAUUSD_MTF_BAR_EXPORTER_P8A1.mq5",
    "mt5_ea/XAUUSD_STRATEGY_PORTFOLIO_LIVE_V02.mq5",

    "scripts/xauusd_multi_strategy_engine_v01.py",
]

errors = []

for rel in required:
    p = ROOT / rel

    if not p.exists():
        errors.append(
            f"MISSING: {rel}"
        )

cfg_path = (
    ROOT
    / "config"
    / "multi_strategy_portfolio.json"
)

if cfg_path.exists():
    try:
        cfg = json.loads(
            cfg_path.read_text(
                encoding="utf-8"
            )
        )

        active = [
            sid
            for sid, strategy
            in cfg["strategies"].items()
            if strategy.get("enabled")
        ]

        print(
            "Active strategies:",
            active,
        )

    except Exception as exc:
        errors.append(
            f"CONFIG ERROR: {exc}"
        )

print()

if errors:
    print("RUNTIME VALIDATION = FAIL")

    for error in errors:
        print(error)

    raise SystemExit(1)

print("RUNTIME VALIDATION = PASS")
