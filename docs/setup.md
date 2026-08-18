# Setup Guide

## Quick Start

```bash
cd D:/StockTradingBot
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Configure Robinhood MCP
cp .env.example .env
# Edit .env with your MCP credentials

# Run dry-run to verify setup
python main.py dry-run
```

## Robinhood MCP Authentication

See `references/robinhood/mcp-setup-guide.md` for detailed connection steps.

Quick connect options:

### Claude Code
```bash
claude mcp add robinhood-trading --transport http https://agent.robinhood.com/mcp/trading
```

### Claude Desktop
Settings → Connectors → Add custom connector → `https://agent.robinhood.com/mcp/trading`

### Other Platforms
All support the same URL: `https://agent.robinhood.com/mcp/trading`

After connecting, your `.env` should have the auth token/header extracted from your platform's session. Or use the stdio proxy approach described in `references/robinhood/mcp-setup-guide.md`.

## Run Modes

| Command | Description | Needs Auth |
|---------|-------------|------------|
| `python main.py dry-run` | Engine runs with MockBroker, no market data needed | No |
| `python main.py live --strategy ema_cross_rsi` | Live trading via Robinhood MCP, confirms strategy first | Yes |
| `python main.py backtest --symbols AAPL --start 2022-01-01` | Historical backtest using yfinance | No |
| `python main.py sentiment AAPL` | Quick sentiment score via StockTwits+Reddit | No |
| `python main.py ui` | Launch Streamlit dashboard at localhost:8501 | Optional |

## Troubleshooting

See `troubleshooting.md` for common issues.
