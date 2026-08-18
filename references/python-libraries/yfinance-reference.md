# yfinance Reference

**Package**: `yfinance >= 0.2.30`  
**GitHub**: https://github.com/ranaroussi/yfinance

## Key Functions

### Download Historical Data
```python
import yfinance as yf
df = yf.download('AAPL', start='2022-01-01', end='2024-01-01', 
                 interval='1d', auto_adjust=True, progress=False)
```

Valid intervals: `1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo`

### Ticker Object (for fundamentals/info)
```python
ticker = yf.Ticker('AAPL')
info = ticker.info          # PE, market_cap, sector, etc.
fast_info = ticker.fast_info  # lighter-weight info
earnings = ticker.earnings_dates  # earnings calendar
dividends = ticker.dividends  # dividend history
splits = ticker.splits        # stock split history
```

### Interval Limits (Important!)
- **1m**: max 30 days lookback; split into 7-day chunks per request
- **2m through 60m**: max 60 days lookback
- **1d and above**: unlimited lookback

### Common Issues
- **429 Rate Limited**: Yahoo imposes rate limits. Add delays between calls.
- **Missing Dividend Adjustment**: `auto_adjust=True` handles splits/dividends. Without it, price gaps appear at adjustment dates.
- **MultiIndex Columns**: When fetching multiple tickers simultaneously, columns become MultiIndex (Ticker, column_name). Flatten with `df.columns = df.columns.get_level_values(0)`.

### Our Usage
Used as fallback datasource in `bot/plugins/datasources/yfinance_source.py`. Handles chunking for 1m data automatically.
