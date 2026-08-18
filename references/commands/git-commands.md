# Git Commands Reference

This project follows a linear workflow with frequent commits during development.

## Repository Setup

```bash
cd D:/StockTradingBot
git init
git branch -M main
gh repo create StockTradingBot --private --source=. --remote=origin
```

Repo: https://github.com/phoenixfire808/StockTradingBot (private)

## Commit Cadence

One logical commit per completed feature/module. Message format:
```
Step N: <description>

Examples:
Step 1: scaffold repo with plugin architecture
Step 2: add core framework (registry, auto-discovery)
Step 3: implement yfinance + robinhood_mcp datasources
Step 4: EMA cross RSI strategy plugin
Step 5: sentiment module with StockTwits + Reddit sources
Step 6: broker abstraction (MockBroker, RobinhoodMcpBroker)
Step 7: live engine with full account management
Step 8: CLI entry point with all subcommands
Step 9: Streamlit dashboard UI
Step 10: test suite
Step 11: docs, references vault, tracking files
Final: complete scaffold + source code committed
```

## Common Commands

```bash
git status                    # Check staged changes
git diff --stat               # Quick overview of changes
git add -A                    # Stage all changes
git commit -m "commit message"
git push origin main          # Push to remote
git log --oneline             # View recent history
```

## Branch Strategy

Linear only — no feature branches planned during initial build. Once the framework is stable, use `feature/<name>` for new plugins/capabilities.

## Safe Directory (Windows)

If you see "dubious ownership" errors:
```bash
git config --global --add safe.directory D:/StockTradingBot
```
