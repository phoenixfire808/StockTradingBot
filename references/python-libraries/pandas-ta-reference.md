# pandas_ta Reference

**Package**: `pandas-ta >= 0.4.4b0` (for Python 3.13+)  
**GitHub**: https://github.com/twopirllc/pandas-ta

## Key Indicators Used

```python
import pandas_ta as ta

# Exponential Moving Average
ema9 = ta.ema(close, length=9)

# Relative Strength Index
rsi14 = ta.rsi(close, length=14)

# Average True Range
atr14 = ta.atr(high, low, close, length=14)

# Bollinger Bands
bb = ta.bbands(close, length=20, std=2)
# Returns DataFrame with columns: BBEL_20_2, BBBL_20_2, BBDM_20_2
# (upper band, lower band, middle band)

# MACD
macd = ta.macd(close, fast=12, slow=26)

# Stochastic RSI
stochrsi = ta.stochrsi(close, length=14)
```

## Python 3.13 Compatibility Note

- **pandas-ta 0.3.x**: Broken on numpy>=2 (removed `np.NaN`). DO NOT USE with Python 3.13 + numpy 2+.
- **pandas-ta 0.4.4b0+**: Supports numpy 2+ and pandas 2+. RECOMMENDED for Python 3.13.

Our `bot/indicators.py` attempts `import pandas_ta` first; if it fails, falls back to pure-pandas manual formulas (EMA via `ewm()`, RSI via Wilder smoothing, ATR via rolling true range). Both paths produce identical function signatures.

## Fallback Manual Formulas (no external deps)

```python
# EMA
ema = series.ewm(span=period, adjust=False).mean()

# RSI (Wilder smoothing)
delta = series.diff()
gain = delta.where(delta > 0, 0.0)
loss = (-delta).where(delta < 0, 0.0)
avg_gain = gain.rolling(window=period, min_periods=period).mean()
avg_loss = loss.rolling(window=period, min_periods=period).mean()
rs = avg_gain / avg_loss.replace(0, float('inf'))
rsi = 100 - (100 / (1 + rs))

# ATR (True Range average)
high = df['High']; low = df['Low']
prev_close = df['Close'].shift(1)
tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
atr = tr.rolling(window=period).mean()

# Bollinger Bands
mid = series.rolling(period).mean()
std_dev = series.rolling(period).std()
upper = mid + num_std * std_dev
lower = mid - num_std * std_dev
```
