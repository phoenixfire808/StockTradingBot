# Architecture Overview

## Module Diagram
```
main.py → bot/ (config, data, strategy, risk, broker, engine, backtest, sentiment)
  ├── bot/core/registry.py    — Plugin registry (register/get/all)
  ├── bot/core/plugins.py     — Auto-discovery scan of bot/plugins/{kind}/
  ├── bot/config.py           — Settings dataclass + load_settings()
  ├── bot/data.py             — DataHub: fetch_history via plugin priority order + CSV cache
  ├── bot/indicators.py       — EMA, RSI, ATR, Bollinger (pandas-ta with manual fallback)
  ├── bot/strategy.py         — Strategy ABC + EmaCrossRsi starter
  ├── bot/risk.py             — position_size(), stop_loss(), take_profit(), KillSwitch
  ├── bot/broker.py           — Broker ABC → RobinhoodMcpBroker / MockBroker
  ├── bot/engine.py           — Live loop: reconcile positions → manage exits → evaluate signals → persist state
  ├── bot/backtest.py         — Backtesting runner using backtesting.py
  ├── bot/sentiment.py        — SentimentEngine aggregates sentiment_source plugins
  └── bot/plugins/            — Drop-in modules: strategies/, datasources/, sentiment_sources/
ui/app.py → ui/pages/ (6 Streamlit pages)
docs/ → architecture, setup, troubleshooting
references/ → API docs, tool tables, setup guides (research vault)
```

## How Plugins Work

Every plugin is a Python file in `bot/plugins/{kind}/` that exposes a single module-level attribute named `plugin`:

### Strategy Plugin
```python
# bot/plugins/strategies/my_strategy.py
from bot.strategy import Strategy

class MyStrategy(Strategy):
    name = "my_strategy"
    params = {"param_a": 10}
    
    def generate_signals(self, df): ...

plugin = MyStrategy()  # ← this line is required for auto-discovery
```

The core scans `bot/plugins/strategies/*.py`, imports each module, reads `module.plugin`, calls `STRATEGIES.register(plugin.name, plugin)`. Import errors are silently logged — one broken plugin never breaks the system.

### Data Source Plugin
```python
# bot/plugins/datasources/my_datasource.py
class MySource:
    name = "my_source"
    priority = 5  # lower = tried first
    
    def supports(self, interval: str) -> bool: ...
    def fetch_history(self, symbol, start, end, interval): ...

plugin = MySource()
```

DataHub tries datasources sorted by priority. On failure, it falls through to the next. All sources must return a DataFrame with columns Open/High/Low/Close/Volume and a DatetimeIndex.

### Sentiment Source Plugin
```python
# bot/plugins/sentiment_sources/my_sentiment.py
class MySentiment:
    name = "my_sentiment"
    
    def fetch(self, symbol: str, limit=30) -> list[RawPost]: ...

plugin = MySentiment()
```

RawPost has fields: body (str), created_at (str). The SentimentEngine scores them with VADER.

## Adding a New Strategy (Recipe)

1. Create `bot/plugins/strategies/new_strategy.py`
2. Subclass `Strategy` from `bot/strategy`
3. Set `name`, override `generate_signals()` returning an int8 Series (1/-1/0)
4. Expose `plugin = YourClass()` at module level
5. Restart — the new strategy auto-appears everywhere (CLI, UI, backtests)

## How the Engine Manages the Account

Each cycle runs in order:
1. **Kill switch check** — daily loss guard + `logs/kill_switch.flag`
2. **Position reconciliation** — adopt external positions, drop stale ones
3. **Exit management** — stop/target hit, or signal exit → sell
4. **Order cleanup** — cancel unmatched open orders older than one cycle
5. **Entry signals** — positive signal on tracked symbols → buy
6. **Persist** — write trades.csv, equity_history.csv, positions_state.json

## Risk Controls

| Control | Description | Who Sets It |
|---------|-------------|-------------|
| Position size | Max `(equity * risk_per_trade) / stop_distance`, capped at 25% equity | config `risk_per_trade` |
| Stop-loss | Entry - 2×ATR | computed per trade |
| Take-profit | Entry + 3×ATR (1.5:1 R:R) | computed per trade |
| Daily loss kill switch | Halt if drawdown > `max_daily_loss_pct` from day start | config |
| Emergency stop | `logs/kill_switch.flag` written by UI button | user action |

Blast radius is limited to Robinhood Agentic account — writes can only occur within that sub-account, which only contains funds the user moves into it.
