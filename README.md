# XAUUSD Multi-Strategy ML Portfolio Bot

Extensible MetaTrader 5 + Python framework for running independent
machine-learning trading strategies on XAUUSD.

## Current strategies

### V03H_F002_SELL

- M15
- SELL
- Random Forest / ONNX
- 300 locked features
- percentile-rank decision
- current operational rank threshold: 0.90
- SL: 1 ATR
- TP: 2 ATR
- max hold: 32 M15 bars
- magic: 7300201

### V05H_BUY

- M5
- BUY
- Random Forest / ONNX
- 830 locked features
- probability decision
- cutoff: 0.9391089677810669
- SL: 1.5 ATR
- TP: 2 ATR
- max hold: 6 M5 bars
- magic: 7300501

The strategies execute independently. One model does not veto another.

## Architecture

    MetaTrader 5
         |
         v
    P8A1 MTF Bar Exporter
         |
         +-- M5
         +-- M15
         +-- H1
         +-- H4
         +-- D1
         |
         v
    Python feature pipelines
         |
         +--> V03H SELL
         |
         +--> V05H BUY
         |
         v
    Multi-Strategy Engine
         |
         v
    portfolio/inbox/*.signal
         |
         v
    Portfolio LIVE V02 EA
         |
         v
    Independent MT5 positions

## Requirements

- Windows with MetaTrader 5
- WSL2 Ubuntu recommended for Python runtime
- Python 3
- MT5 hedging account strongly recommended if simultaneous BUY and
  SELL positions should remain independent

## Installation

Clone this repository:

    git clone <YOUR_GITHUB_REPOSITORY>
    cd xauusd-multi-strategy-ml-portfolio

Create an environment:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

Download the models from Hugging Face:

    python3 scripts/download_models.py

The configured Hugging Face account is:

    addyAIMLprojects

Model repositories:

    addyAIMLprojects/xauusd-v03h-sell
    addyAIMLprojects/xauusd-v05h-buy

For private model repositories, authenticate first:

    hf auth login

Validate:

    python3 scripts/validate_runtime.py

## MetaTrader setup

Compile these in MetaEditor:

    mt5_ea/XAUUSD_MTF_BAR_EXPORTER_P8A1.mq5
    mt5_ea/XAUUSD_STRATEGY_PORTFOLIO_LIVE_V02.mq5

Attach the P8A1 exporter to the broker's XAUUSD symbol.

Attach Portfolio LIVE V02 to one XAUUSD chart.

Do not run the legacy P7 or Portfolio V01 executor simultaneously.

For first testing:

    InpExecutionEnabled = false

Then run Python:

    python3 scripts/xauusd_multi_strategy_engine_v01.py       --config config/multi_strategy_portfolio.json

## MT5 Common Files

The runtime attempts automatic discovery.

To override it:

    export XAU_MT5_COMMON_FILES="/mnt/c/Users/YOURUSER/AppData/Roaming/MetaQuotes/Terminal/Common/Files"

Expected runtime structure:

    Common/Files/xau_signals/
      xauusd_m5_bars.csv
      xauusd_m15_bars.csv
      xauusd_h1_bars.csv
      xauusd_h4_bars.csv
      xauusd_d1_bars.csv
      portfolio/
        inbox/
        processed/
        rejected/
        expired/
        queue.txt
        strategy_contracts.csv

## Adding future strategies

Strategies are registered in:

    config/multi_strategy_portfolio.json

Each strategy has its own:

- feature builder
- model
- timeframe
- direction
- decision rule
- threshold
- ATR trade contract
- maximum hold
- unique magic number

The execution framework is designed so future strategies do not need
to be hardcoded by model name.

## Important

The included models depend on exact feature contracts. Do not reorder
or substitute model features without validating or retraining the model.

Market timestamps must use correct UTC semantics.

## Risk notice

This is experimental trading software.

Historical or backtested performance does not guarantee future
performance. Leveraged trading can result in substantial losses.

Test on a demo account before any live deployment.
