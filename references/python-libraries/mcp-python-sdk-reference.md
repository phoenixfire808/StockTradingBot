# MCP Python SDK Reference

**Package**: `mcp >= 1.0.0`  
**GitHub**: https://github.com/modelcontextprotocol/python-sdk

## Transport Options

### SSE (Streamable HTTP) — for remote servers like Robinhood MCP
```python
from mcp import ClientSession
from mcp.client.sse import sse_client

async with sse_client(url="https://agent.robinhood.com/mcp/trading", 
                      headers={"Authorization": "Bearer TOKEN"}) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        
        # List available tools
        tools = await session.list_tools()
        
        # Call a tool
        result = await session.call_tool("get_portfolio")
        # Result.content is a list of content blocks
        
        # Send prompts or resources
        prompts = await session.list_prompts()
        resources = await session.list_resources()
```

### Stdio — for local subprocess servers
```python
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters

params = StdioServerParameters(
    command="python",
    args=["path/to/server.py"],
)

async with stdio_client(params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
```

## Calling Tools — Result Parsing

Tool results come as a list of content blocks. Each block may have `.text` attribute or be a dict with a `text` key. Parse by extracting all `.text` fields and joining, then attempt JSON parse:

```python
result = await session.call_tool("get_equity_quotes", {"symbols": ["AAPL", "MSFT"]})
content_parts = []
for block in result.content:
    if hasattr(block, 'text'):
        content_parts.append(block.text)
    elif isinstance(block, dict) and 'text' in block:
        content_parts.append(block['text'])
    elif isinstance(block, str):
        content_parts.append(block)
text_result = "\n".join(content_parts)
data = json.loads(text_result)  # try parsing as JSON
```

## Important Notes

- **Session lifecycle**: The session must stay alive for the duration of tool calls. Create one session at startup and keep it open.
- **Authentication**: Robinhood MCP uses platform OAuth. The token comes from your AI platform session after you authenticate. There is no standalone API key for the MCP endpoint.
- **Error handling**: If the MCP server rejects auth (wrong token, expired), it closes the connection. Catch and retry once with re-authentication.
- **Timeout**: Default timeouts vary; set explicit timeouts for production use.

## Our Implementation
See `bot/broker.py` → `RobinhoodMcpBroker` class. Supports both SSE remote and local stdio proxy modes based on `.env` config.
