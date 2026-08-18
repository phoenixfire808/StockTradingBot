# Equities Tools Reference

Source: https://robinhood.com/us/en/support/articles/trading-with-your-agent/

## Equity Tools

| Tool | Description |
|------|-------------|
| `get_equity_positions` | View open equity positions with quantity and cost basis |
| `get_equity_tax_lots` | View open tax lots with quantity, cost basis, acquisition date, long/short status |
| `get_equity_quotes` | Get real-time equity quotes and prior close for up to 20 symbols |
| `get_equity_orders` | Get equity order status history |
| `get_equity_tradability` | Check if a symbol can be traded and if fractional trading is available |
| `review_equity_order` | Simulate an equity order and get pre-trade warnings |
| `place_equity_order` | Place a real equity order |
| `cancel_equity_order` | Cancel an open equity order |

## Order Types Supported

Robinhood supports market, limit, stop, and stop-limit orders through the API. Bracket orders and OCO orders are NOT natively supported — manage stop/target logic in the engine instead.

## Implementation Notes

- `place_equity_order` takes: `symbol`, `qty`, `side` (buy/sell), `type` (market/limit/stop/stop_limit), plus optional `limit_price`, `stop_price`, `time_in_force`
- `get_equity_quotes` accepts comma-separated symbols, returns real-time bid/ask/mark
- `review_equity_order` returns warning/error objects without placing anything — use before placing risky orders
