# Social Sentiment Sources Reference

## StockTwits

### Endpoint
```
GET https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json?limit=30
```

### Headers
```
User-Agent: StockTradingBot/1.0 (research)
```

### Auth
No API key required for basic read access. Optional bearer token unlocks higher rate limits.

### Rate Limits
~30 requests/hour per IP without authentication. With app credentials: significantly higher.

### Response Format
```json
{
  "messages": [
    {
      "id": 823847562,
      "body": "$AAPL breaking resistance at $200 🚀",
      "created_at": "2026-08-18T15:30:00Z",
      "user": {
        "username": "bullrunner",
        "avatar_url": "..."
      },
      "symbols": [{"symbol": "AAPL", "ticker": "AAPL"}],
      "mentioned_users": [],
      "entities": {},
      "likes": {"total": 12, "user_ids": []}
    }
  ]
}
```

### CAShtag Handling
Messages often start with a ticker symbol followed by `$SYMBOL` (cashtag). Strip cashtags before VADER scoring if desired, though our implementation scores the full body text which captures contextual sentiment around the ticker mention.

### Error Responses
- `404`: Symbol not found / no messages
- `429`: Rate limited — wait 30s+ between requests
- Other errors: log and skip

### File Location
Implemented in `bot/plugins/sentiment_sources/stocktwits.py`

## Reddit r/wallstreetbets

### Endpoint
```
GET https://www.reddit.com/r/wallstreetbets/new.json?limit=100
```

### Headers
```
User-Agent: StockTradingBot/1.0 (research)
```

### Auth
Public JSON endpoint — no OAuth required. However, Reddit aggressively blocks requests with generic User-Agent strings. Must use a descriptive UA.

### Rate Limits
~60 requests/minute with valid UA. After excessive requests: 429 with retry-after header.

### Response Structure (simplified)
```json
{
  "data": {
    "children": [
      {
        "data": {
          "title": "NVDA moon bound 🌕",
          "selftext": "Just looking at the fundamentals...",
          "created_utc": 1724000000,
          "link_flair_text": null
        }
      }
    ]
  }
}
```

### Ticker Matching
Posts are matched by searching title + selftext for:
- `$SYMBOL` (cashtag pattern, e.g., `$AAPL`)
- Bare ticker word-boundary match (e.g., `AAPL`)

Both patterns are case-insensitive word-boundary regex matches.

### Error Responses
- `429`: Rate limited — use exponential backoff
- `403`: Banned (generic UA) — always send descriptive UA
- Network errors: log and skip

### File Location
Implemented in `bot/plugins/sentiment_sources/reddit.py`

## VADER Sentiment Scoring

All posts scored using the `vaderSentiment` library's compound score:
- **Bullish**: compound > 0.25
- **Bearish**: compound < -0.25
- **Neutral**: -0.25 ≤ compound ≤ 0.25

Compound ranges from -1 (extreme negative) to +1 (extreme positive).

VADER was specifically designed for social media text and performs well without domain-specific training. The lexicon is self-contained — no download or network dependency at runtime.

### File Location
Scoring engine: `bot/sentiment.py` via `vaderSentiment.SentimentIntensityAnalyzer().polarity_scores(text)["compound"]`
