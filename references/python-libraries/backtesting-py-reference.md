# backtesting.py Reference

**Package**: `backtesting >= 0.3.3`  
**Docs**: https://kernc.github.io/backtesting.py/

## Core Classes

### Backtest
```python
from backtesting import Backtest, Strategy

bt = Backtest(
    df,              # DataFrame with OHLCV columns
    MyStrategy,      # Strategy subclass
    cash=100_000,    # Starting capital
    commission=0.0005,  # Commission rate (0.05%)
    exclusive_orders=True,  # One position at a time
)
```

### Strategy Base Class
```python
class MyStrategy(Strategy):
    def init(self):
        # Precompute indicators here (use self.SMA(), self.I() etc.)
        pass
    
    def next(self):
        # Decision logic executed once per bar
        if not self.position and condition_to_buy:
            self.buy()
        elif self.position and condition_to_sell:
            self.close()
```

Common built-in indicators in Strategy: `self.SMA(data, n)`, `self.I(func, *args)` for custom functions.

### Running & Metrics
```python
stats = bt.run()
# Returns dict with keys:
#   'Return [%]', 'Buy & Hold Return [%]', 'Return [Avg./Ann.] [%]',
#   'Drawdown [%]', 'Max Drawdown [%]', 'Position Coverage [%]',
#   '# Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]',
#   'Avg. Winning Trade [%]', 'Avg. Losing Trade [%]',
#   'Avg. Duration [bars]', 'Profit Factor', 'Expectancy [%]',
#   'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Omega Ratio',
#   'SQN', 'Kelly Criterion'
```

### Plotting
```python
bt.plot(filename='report.html', open_browser=False, 
        grid=True, volume=True, show_legend=True)
```

### Accessing Trades
```python
trades = bt.trades()
# Returns DataFrame: Entry_Time, Exit_Time, Size, Entry_Price,
# Exit_Price, High_i, Low_i, PnL, Return%, Duration, Tag
```

## Our Usage

In `bot/backtest.py` we use backtesting.py purely for historical validation of strategy rules. Live trading uses the MCP broker directly. The `Strategy.to_backtesting_strategy()` method converts our abstract strategy class into a backtesting-compatible format with zero duplicated signal logic.
