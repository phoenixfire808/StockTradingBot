# Sources Index

Master index of all data sources, APIs, and research materials for this project.

## Robinhood

| Source | URL | Date Accessed | Coverage |
|--------|-----|---------------|----------|
| Agentic Trading Overview | https://robinhood.com/us/en/support/articles/agentic-trading-overview/ | 2026-08-18 | Account model, auth flow, agent capabilities, safety controls |
| Trading with Your Agent | https://robinhood.com/us/en/support/articles/trading-with-your-agent/ | 2026-08-18 | Tool table (equities, market data, portfolio, watchlist, options, crypto, scanner) |
| MCP Setup Guide | references/robinhood/mcp-setup-guide.md | - | Platform-specific connection steps |
| Equities Tools Reference | references/robinhood/equities-tools.md | - | place_equity_order, cancel_equity_order, review_equity_order, get_equity_quotes, etc. |
| Market Data Tools Reference | references/robinhood/market-data-tools.md | - | get_equity_historicals, fundamentals, technical_indicators, etc. |
| Portfolio Tools Reference | references/robinhood/portfolio-tools.md | - | get_accounts, get_portfolio, get_realized_pnl, search |

## External Libraries

| Library | Purpose | Key Feature |
|---------|---------|-------------|
| mcp >= 1.0.0 | Robinhood MCP client | SSE + stdio transport |
| backtesting >= 0.3.3 | Strategy backtesting | Event-driven, single API |
| yfinance >= 0.2.30 | Free OHLCV data | Supports intervals from 1m to 3mo |
| pandas-ta >= 0.4.4b0 | Technical indicators | EMA, RSI, ATR, Bollinger Bands |
| streamlit >= 1.36.0 | Dashboard UI | Multi-page app, dark theme support |
| plotly >= 5.22.0 | Charts | Candlesticks, subplots, gauges |
| vaderSentiment >= 3.3.2 | Sentiment scoring | VADER lexicon, self-contained |
| APScheduler >= 3.10.4 | Engine scheduling | Blocking scheduler for live loop |
| requests >= 2.31.0 | Web crawler | HTTP for sentiment data sources |

## Social Sentiment Sources

| Source | Endpoint | Auth Required | Rate Limit |
|--------|----------|---------------|------------|
| StockTwits | api.stocktwits.com/api/2/streams/symbol/{symbol}.json | No | ~30 req/hour per IP |
| Reddit r/wallstreetbets | www.reddit.com/r/wallstreetbets/new.json | No (needs UA header) | ~60 req/min with valid UA |

## Tick Data Options (Future Plugin Candidates)

| Provider | Type | Cost | Notes |
|----------|------|------|-------|
| yfinance intraday | Bar-level (1m min) | Free | 1m: ≤30d lookback; 7d per request chunk |
| Polygon.io | Tick + bars | Paid ($20+/mo) | Real-time ticks, US equities |
| Databento | Tick + bars | Pay-per-use | Ultra-low latency, institutional-grade |
| Alpaca (data-only) | Bars + some ticks | Free tier | Not used as broker, but data endpoint available |
