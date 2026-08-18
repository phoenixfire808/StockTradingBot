# Robinhood Agentic Trading Overview

**Source**: https://robinhood.com/us/en/support/articles/agentic-trading-overview/

Launched May 27, 2026, Robinhood became the first major retail broker to expose live equity execution to AI agents via an official Model Context Protocol (MCP) server.

## Account Model

- **Dedicated Agentic account**: A separate, user-funded sub-account. The agent can trade against it, but the main portfolio is never directly reachable. Functions like a trading allowance — move in what you're comfortable letting the agent control.
- **Read access across ALL accounts**: positions, balances, portfolio history, open orders, transactions, watchlists.
- **Write access scoped ONLY to Agentic account**: the agent can only execute trades in this dedicated sub-account.

## Safety Controls

| Control | Description |
|---------|-------------|
| Per-trade push notifications | Every order fires a real-time user alert |
| Optional approval gate | Users CAN require manual confirmation before each trade (toggle in Robinhood app) |
| Fraud detection | Suspicious activity flags a Robinhood team review |
| Monthly spending limits | On the companion virtual agentic credit card |
| Access revocation | Disconnect agent access at any time |

## What Your Agent Can Do

- Build portfolios ("Look through news and industry reports to build a portfolio that represents little-known tickers across the AI supply chain.")
- Automate trading strategies ("Buy $100 of ROAR every time the price decreases 2% or more in 1 day.")
- Adjust your portfolio ("Rebalance my portfolio to achieve a 20% allocation in ROAR and 80% allocation in HMNI.")
- Analyze your portfolio ("Look at my portfolio and tell me what risks I'm exposed to.")
- Analyze market data ("Why is ROAR up today?" / "Look at news, social sentiment, and recent quotes to build a bull and bear thesis for ROAR.")

## Supported Instruments (at beta)

- Equities (launched)
- Crypto (available if Robinhood Crypto account exists)
- Options (signalled to follow)
- Futures, event contracts, prediction markets (future)

## State Restrictions

Crypto trading through your agent isn't available in every state (including New York). If you move to a restricted state, your agent loses ability to place new crypto trades. Existing orders are not cancelled.

## Disclosures

> Agentic trading involves significant risk, including the possible loss of your entire investment. AI-driven strategies may perform poorly under certain market conditions, move quickly, and may be difficult to monitor or stop in real time. You assume all risk for trades executed by AI agents.

## Connecting Your AI Agent

Use the MCP link: `https://agent.robinhood.com/mcp/trading`

Supported platforms: Claude Code, Claude Desktop, ChatGPT, OpenAI Codex/Codex CLI, Cursor, Grok. See `references/robinhood/mcp-setup-guide.md` for step-by-step on each platform.
