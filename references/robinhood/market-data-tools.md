# Market Data Tools Reference

Source: https://robinhood.com/us/en/support/articles/trading-with-your-agent/

## Market Data Tools

| Tool | Description |
|------|-------------|
| `get_equity_historicals` | Get OHLCV price bars across a time range |
| `get_equity_fundamentals` | Get valuation ratios, market cap, 52-week range, dividend info, and today's OHLCV |
| `get_financials` | Get a company's reported financials over time — revenue, gross profit, net income, net margin by quarter or year |
| `get_equity_price_book` | Get real-time Level 2 order book showing bid/ask levels and resting size, up to 4 stocks |
| `get_equity_technical_indicators` | Compute a technical indicator (RSI, MACD, Bollinger Bands, moving averages, more) for a stock over a time range |
| `get_earnings_results` | Look up a specific stock's earnings history and next report |
| `get_earnings_calendar` | List earnings reports scheduled across the market over a date window (up to 31 days) |
| `get_indexes` | Look up market indexes by symbol |
| `get_index_quotes` | Get real-time index values |

## Technical Indicators Available

Per Robinhood docs: RSI, MACD, Bollinger Bands, Moving Averages, and more. This could be an alternative data source for indicators if needed.

## Historical Bars Granularity

The exact interval parameters supported by `get_equity_historicals` were confirmed but variable — implement should query tool schema at first use and log discovered intervals (as noted in `bot/plugins/datasources/robinhood_mcp_source.py`).
