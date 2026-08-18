# Troubleshooting

## MCP Connection Fails
- Verify credentials in `.env` match what your AI platform authenticated.
- Test with `python main.py live` — if `test_connection` returns False, check network and credential validity.
- Dry-run always works: `python main.py dry-run` uses MockBroker (no MCP needed).

## pandas-ta Import Fails
- If `pandas_ta` is incompatible with your Python/numpy version, indicators fall back to manual pure-pandas implementations automatically. Check logs for the path taken.

## yfinance Returns 429 Rate Limited
- yfinance limits requests. Use cached data (`data/` directory) — already handles staleness (refreshes after 1 day).
- For frequent requests, increase lookback intervals (e.g., `interval="1h"` instead of `"1d"` won't help; try longer periods between runs).

## Streamlit Port Already in Use
- Use `python main.py ui --port 8502` or any free port.

## Plugin Not Discovering
- Ensure the file has a `plugin = ClassName()` module-level attribute.
- Check logs for import errors — bad plugins are skipped gracefully.

## Kill Switch Tripping Frequently
- Reduce `MAX_DAILY_LOSS_PCT` in `.env` to a less sensitive value (default 3%).
- Check the EMERGENCY STOP button was accidentally pressed.
- Delete `logs/kill_switch.flag` to re-arm manually.
