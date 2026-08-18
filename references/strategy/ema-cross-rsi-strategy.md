# EMA Cross + RSI Filter Strategy

## Overview

A trend-following mean-reversion hybrid strategy that enters on moving average crossovers but only when the asset hasn't become overbought (via RSI filter). Designed as the first production-ready plugin for the modular framework.

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `fast` | 9 | 5–20 | Fast EMA period |
| `slow` | 21 | 15–50 | Slow EMA period |
| `rsi_period` | 14 | 7–21 | RSI calculation period |
| `rsi_entry_max` | 70.0 | 60–80 | Max RSI for long entry (below this = not overbought) |
| `rsi_exit` | 75.0 | 60–85 | RSI above this triggers exit regardless of crossover |

## Entry Logic

**Long when ALL conditions met:**
1. EMA(fast) crosses **above** EMA(slow) — confirmed by `(fast > slow) & (fast.shift(1) <= slow.shift(1))`
2. RSI(period) < rsi_entry_max — the trend is real, not an overextended spike

```python
cross_up = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
valid_entry = cross_up & (rsi_vals < rsi_entry_max)
```

## Exit Logic

**Exit when ANY condition met:**
1. EMA(fast) crosses **below** EMA(slow) — trend has reversed
2. RSI(period) > rsi_exit — asset is overbought, take profits early

```python
exit_signal = (ema_fast < ema_slow & ema_fast.shift(1) >= ema_slow.shift(1)) | (rsi_vals > rsi_exit)
```

## Signal Output

Returns an int8 Pandas Series aligned to the input DataFrame index:
- `1` = Go long (buy signal)
- `-1` = Exit position (sell signal)  
- `0` = Hold / flat

## Risk Integration

When a buy signal fires:
- Position size = `(equity * risk_per_trade) / (2 * ATR)` capped at 25% of equity
- Stop-loss = `entry_price - 2 * ATR`
- Take-profit = `entry_price + 3 * ATR` (1.5:1 risk/reward)

## Why This Strategy

- **Trend following**: EMA crossover catches directional moves
- **Overbought protection**: RSI filter prevents buying tops during extended rallies
- **Fast signals**: Shorter EMA periods (9/21) react quickly to trend changes
- **Proven concept**: Moving average crossover is the most studied simple strategy; RSI filter is a well-documented enhancement
- **Configurable**: All parameters tunable for different timeframes/volatility environments

## Limitations

- Whipsaws in sideways/choppy markets (use daily bars minimum)
- Lagging indicator — enters after trend has already started
- Single timeframe — doesn't consider higher-timeframe context
- No volume confirmation — ignores supply/demand dynamics

For advanced users, extend the plugin framework with momentum scanners, volume filters, or multi-timeframe analysis.
