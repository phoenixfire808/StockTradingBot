# CLI Commands Reference

## `python main.py backtest`

Run historical backtests on configured symbols.

```bash
python main.py backtest --symbols AAPL MSFT NVDA --start 2022-01-01 --end 2024-01-01 --cash 100000 --strategy ema_cross_rsi
```

| Flag | Default | Description |
|------|---------|-------------|
| `--symbols` | From `.env` (AAPL,MSFT,NVDA) | Space-separated ticker symbols |
| `--start` | `2022-01-01` | Start date (YYYY-MM-DD) |
| `--end` | None (today) | End date |
| `--cash` | `100000` | Starting capital |
| `--strategy` | `ema_cross_rsi` | Strategy name from registered plugins |

Output: Formatted metrics table to stdout + `logs/backtest_summary.json` + HTML reports in `reports/`.

## `python main.py dry-run`

Start engine with MockBroker. No authentication or market data needed.

```bash
python main.py dry-run
```

Runs continuous cycle (default 5-minute intervals) using fake prices. Press Ctrl+C to stop. Logs `[MOCK]` markers to console.

## `python main.py live`

Live trading via Robinhood Agentic Trading MCP. One-time strategy confirmation at startup.

```bash
python main.py live --strategy ema_cross_rsi
```

| Flag | Default | Description |
|------|---------|-------------|
| `--strategy` | `ema_cross_rsi` | Strategy name to confirm and run |
| `--symbols` | From `.env` | Override tracked symbols |

Flow:
1. Test MCP connection → exit 2 with setup pointer if failed
2. Display strategy summary (name, params, symbols, risk settings)
3. Prompt: "Trade live on the Robinhood Agentic account with this strategy? [y/N]"
4. On `y`: save confirmation to `logs/strategy_confirmed.json`, start engine
5. Engine runs continuously — manages positions, exits, order lifecycle

No `ALLOW_LIVE` flag needed. No countdown. No refusal gates.

## `python main.py sentiment <symbol> ...`

Quick sentiment score for one or more tickers.

```bash
python main.py sentiment AAPL MSFT --hours 24
```

| Argument | Required | Description |
|----------|----------|-------------|
| `symbols` | Yes | Ticker symbols to check |
| `--hours` | 24 | Lookback window |

Output: Per-symbol mentions count, bullish/bearish counts, net_score (-1..1), top post snippets.

## `python main.py ui [--port 8501]`

Launch Streamlit dashboard.

```bash
python main.py ui
python main.py ui --port 8502
```

Opens http://localhost:8501. Pages: Dashboard, Backtest, Research, Trades, Logs, Settings.

## Configuration (.env)

All config loaded from `.env` (copy from `.env.example`). Keys are optional with defaults:

| Key | Default | Description |
|-----|---------|-------------|
| `ROBINHOOD_MCP_URL` | `https://agent.robinhood.com/mcp/trading` | MCP server URL |
| `ROBINHOOD_MCP_AUTH_HEADER` | Empty | Auth header value (Bearer token) |
| `ROBINHOOD_MCP_COMMAND` | Empty | Local stdio proxy command |
| `ROBINHOOD_MCP_ARGS` | `[]` | JSON array of proxy args |
| `SYMBOLS` | `AAPL,MSFT,NVDA` | Comma-separated symbols |
| `LOG_LEVEL` | `DEBUG` | Logging level |
| `CASH` | `100000` | Default starting cash for backtests |
| `RISK_PER_TRADE` | `0.01` | 1% of equity per trade |
| `MAX_DAILY_LOSS_PCT` | `3.0` | Daily loss kill switch threshold |
| `ENGINE_INTERVAL_MINUTES` | `5` | Engine cycle interval |
| `SENTIMENT_LOOKBACK_HOURS` | `24` | Sentiment analysis lookback |
