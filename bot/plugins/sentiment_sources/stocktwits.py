"""StockTwits public stream API sentiment source plugin.

Endpoint: https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json
No API key required for basic read access. Rate-limited by StockTwits.
"""

import logging
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


@dataclass
class RawPost:
    body: str
    created_at: str


class StockTwitsSource:
    name = "stocktwits"

    def fetch(self, symbol: str, limit: int = 30) -> list[RawPost]:
        """Fetch recent messages for a ticker from StockTwits."""
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        headers = {"User-Agent": "StockTradingBot/1.0 (research)"}
        params = {"limit": min(limit, 30)}

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            messages = data.get("messages", [])
            posts = []
            for msg in messages[:limit]:
                body = msg.get("body", "")
                created_at = msg.get("created_at", "")
                posts.append(RawPost(body=body, created_at=created_at))
            return posts
        except requests.exceptions.HTTPError as e:
            if resp.status_code in (404, 429):
                logger.warning("StockTwits %s for %s: rate limit or not found", resp.status_code, symbol)
                return []
            logger.warning("StockTwits HTTP error for %s: %s", symbol, e)
            return []
        except Exception as e:
            logger.warning("StockTwits fetch failed for %s: %s", symbol, e)
            return []
        finally:
            time.sleep(1.0)  # polite rate limiting between requests
