#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

python3 scripts/xauusd_multi_strategy_engine_v01.py \
    --config config/multi_strategy_portfolio.json
