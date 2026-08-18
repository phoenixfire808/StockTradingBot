# Tick Data & Intraday Sources Reference

## Free Data (yfinance)

YFinance intraday support varies by interval — hard API limits enforced:

| Interval | Max Lookback | Per-Request Limit | Chunking Needed |
|----------|-------------|-------------------|-----------------|
| 1m | 30 days | 7 days per request | Yes — split into 7-day windows |
| 2m | 60 days | N/A | No |
| 5m | 60 days | N/A | No |
| 15m | 60 days | N/A | No |
| 30m | 60 days | N/A | No |
| 60m | 60 days | N/A | No |
| 1h | Same as 60m | Same | No |
| 1d+ | Unlimited | N/A | No |

**Limitations**: No true tick-level data. 1-minute bars are the finest granularity available free. The data is delayed ~15 minutes for most symbols unless using premium Yahoo Finance.

### Chunking Implementation

For 1m data older than 7 days, the `yfinance_source.py` plugin splits into 7-day request windows and concatenates results. See `bot/plugins/datasources/yfinance_source.py`.

## Paid Tick Providers (Future Plugin Candidates)

### Polygon.io
- **URL**: https://polygon.io
- **Pricing**: ~$20–99/month for US equities
- **Data**: Real-time ticks, trades, aggregated bars at any granularity
- **API**: REST + WebSocket; Python SDK `pip install polygon-api-client`
- **Why use it**: True tick-by-tick trade data, Level 1 quotes, real-time streaming
- **Plugin contract**: Would subclass `datasource.PluginBase`, return OHLCV or tick-format DataFrame

### Databento
- **URL**: https://databento.com
- **Pricing**: Pay-per-use (~$10–50/month depending on volume)
- **Data**: Institutional-grade tick data, historical deep dive, NASDAQ CQG/CX
- **API**: REST + gRPC; Python SDK `pip install databento`
- **Why use it**: Best price-quality ratio, microsecond timestamps, exchange-level detail
- **Plugin contract**: Higher bandwidth but lowest cost per record for full market replay

### Alpaca (Data Only)
- **URL**: https://alpaca.markets/data
- **Pricing**: Free tier (100 req/min), Paid tiers from $9/month
- **Data**: Real-time + historical bar data, minute-resolution for free plan
- **Why note it**: We declined as broker but their data API is excellent and well-documented
- **Plugin contract**: Simple REST wrapper, returns clean Pandas DataFrames

## Robinhood MCP Historical Granularity

The Robinhood MCP server's `get_equity_historicals` tool supports interval parameters, but the exact valid values were not fully documented. At implementation time:
1. Run `session.list_tools()` to discover tool schemas
2. Log discovered intervals to `DEBUG_LOG.md`
3. Try common intervals: `"1min", "5min", "15min", "1hour", "1day"`
4. Fall back to whichever works best
