# Robinhood MCP Connection Setup

**MCP Server URL**: `https://agent.robinhood.com/mcp/trading`  
**Transport**: Streamable HTTP (SSE)

## Connect via Claude Code

```bash
claude mcp add robinhood-trading --transport http https://agent.robinhood.com/mcp/trading
# Then type /mcp in Claude Code → select robinhood-trading → authenticate
```

## Connect via Claude Desktop

1. Settings → Connectors → Add custom connector
2. Add MCP link: `https://agent.robinhood.com/mcp/trading`

## Connect via ChatGPT

1. Settings → Security & login → Turn on Developer Mode
2. Plugins → Select + → Add MCP link: `https://agent.robinhood.com/mcp/trading`

## Connect via Codex / Codex CLI

Settings → MCP servers → Select Streamable HTTP → Add: `https://agent.robinhood.com/mcp/trading`

Or via Codex CLI:
```bash
codex mcp add robinhood-trading --url https://agent.robinhood.com/mcp/trading
```

## Connect via Cursor

1. Give this MCP link to your agent: `https://agent.robinhood.com/mcp/trading`
2. Settings → Cursor Settings → Tools & MCPs → Connect

## Connect via Grok

Start a chat → Select + → Add connector → Custom → Add: `https://agent.robinhood.com/mcp/trading`

## Using with Our Bot

The bot connects via the Python MCP SDK. Two modes:

### Remote SSE (recommended)
Set in `.env`:
```
ROBINHOOD_MCP_URL=https://agent.robinhood.com/mcp/trading
ROBINHOOD_MCP_AUTH_HEADER=Bearer YOUR_TOKEN_HERE
```
The auth token is extracted from your AI platform session after authenticating.

### Local Stdio Proxy
Some users prefer running a local authenticated proxy (e.g., the `robinhood-rest2mcp` bridge by Slijeff):
```
ROBINHOOD_MCP_COMMAND=python
ROBINHOOD_MCP_ARGS=["path/to/proxy/server.py"]
```

## Important Notes

- **No API key**: Authentication is OAuth-based, handled by the AI platform. There is no standalone API key.
- **Agentic account only**: Writes are confined to your dedicated Agentic account. Never touches your main holdings.
- **Desktop required**: You must complete the initial onboarding on a desktop device. Mobile devices cannot create an Agentic account.
