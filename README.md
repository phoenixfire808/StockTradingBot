# Stock Trading Bot

A modular plugin-based stock trading bot with live execution via Robinhood Agentic Trading MCP, backtesting engine, social sentiment analysis, and a dark terminal-style Streamlit dashboard.

## Quick Start

```bash
cd D:/StockTradingBot
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials
python main.py dry-run
```

See `docs/setup.md` for full setup instructions.

## Architecture

```
main.py → bot/core/  (registry + auto-discovery)
           ├── config.py     — Settings dataclass + load_settings()
           ├── data.py       — DataHub: priority-ordered datasource plugins + CSV cache
           ├── indicators.py — EMA/RSI/ATR/Bollinger (pandas-ta or manual fallback)
           ├── strategy.py   — Strategy ABC + EmaCrossRsi starter
           ├── risk.py       — Position sizing, stops/targets, daily-loss guard
           ├── broker.py     — Broker ABC → RobinhoodMcpBroker / MockBroker
           ├── engine.py     — Live loop: reconcile → manage exits → evaluate signals → persist
           ├── backtest.py   — Backtesting runner (backtesting.py)
           ├── sentiment.py  — Social sentiment aggregation (VADER scoring)
           └── plugins/      — Drop-in modules: strategies/, datasources/, sentiment_sources/
ui/app.py → 6 Streamlit pages (Dashboard, Backtest, Research, Trades, Logs, Settings)
docs/     — architecture, setup, troubleshooting
references/ — API docs, tool tables, setup guides (research vault)
```

### How Plugins Work

Every plugin is a Python file in `bot/plugins/{kind}/` that exposes a module-level `plugin = ClassName()` attribute. The core auto-discovers them on startup. Adding a feature = dropping one file into the right folder.

See `docs/architecture.md` for detailed plugin contracts and recipe.

## Run Modes

| Command | Description | Auth Needed |
|---------|-------------|-------------|
| `python main.py dry-run` | Engine with MockBroker, no market data | No |
| `python main.py live --strategy ema_cross_rsi` | Live trading via Robinhood MCP | Yes |
| `python main.py backtest --symbols AAPL MSFT` | Historical backtests | No |
| `python main.py sentiment AAPL` | Quick sentiment score | No |
| `python main.py ui` | Dashboard at localhost:8501 | Optional |

Full command reference: `references/commands/cli-commands.md`.

## Disclaimer

⚠️ **This is not financial advice.** Trading stocks involves significant risk including the possible loss of principal. AI-driven strategies may perform poorly under certain conditions. You are solely responsible for your investment decisions. This software is provided "as is" without warranty of any kind. Test thoroughly in dry-run mode before using real capital.

The bot operates within your Robinhood Agentic account — a sandboxed sub-account that only contains funds you move there. Never touches your main holdings.

## Project Structure

```
D:/StockTradingBot/
├── bot/                    # Core trading logic
│   ├── core/               # Plugin registry + auto-discovery
│   ├── plugins/            # Strategies, datasources, sentiment sources
│   ├── config.py, data.py, strategy.py, risk.py, broker.py
│   ├── engine.py, backtest.py, sentiment.py, indicators.py
├── ui/                     # Streamlit dashboard
│   ├── app.py + pages/ (6 pages)
│   └── theme.py
├── docs/                   # Setup, architecture, troubleshooting
├── references/             # Research vault — all APIs, commands, sources
├── tests/                  # Unit tests
├── main.py                 # CLI entry point
├── requirements.txt        # Dependencies
└── .env.example            # Configuration template
```

## Roadmap

See `ROADMAP.md` for planned features and phases.

Live trading ships from day one (Phase P2). The only gate is MCP authentication + one-time strategy confirmation — no per-trade approval required. Risk controls (kill switch, position sizing, stop-losses) remain as safety mechanisms.

## License

MIT
