# Plotly Reference

**Package**: `plotly >= 5.22.0`

## Candlestick Chart
```python
import plotly.graph_objects as go

fig = go.Figure(go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name="Price"
))
# Add overlays:
fig.add_trace(go.Scatter(x=df.index, y=df['ema9'], mode='lines', name='EMA 9', line={'color': '#00C896'}))
fig.show()
```

## Subplots (price + indicator panels)
```python
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.03, row_heights=[0.7, 0.3])
fig.add_trace(go.Candlestick(...), row=1, col=1)
fig.add_trace(go.Bar(y=volume, name="Volume"), row=2, col=1)
fig.update_layout(height=600, title="Symbol Chart")
```

## Gauge Indicator (for sentiment net_score)
```python
fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=net_score,  # -1..1
    gauge={
        'axis': {'range': [-1, 1]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [-1, -0.25], 'color': "#FF5252"},
            {'range': [-0.25, 0.25], 'color': "#888"},
            {'range': [0.25, 1], 'color': "#00C896"},
        ]
    }
))
```

## Line Chart (equity curve)
```python
fig = go.Figure(go.Scatter(x=timestamps, y=equity_values, mode='lines+markers'))
fig.update_layout(title="Equity Over Time", xaxis_title="Date", yaxis_title="Portfolio Value ($)")
```
