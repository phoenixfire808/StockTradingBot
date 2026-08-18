"""Reddit r/wallstreetbets public JSON sentiment source plugin.

Endpoint: https://www.reddit.com/r/wallstreetbets/new.json
Public JSON endpoint — requires a User-Agent header. Rate-limited by Reddit.
"""

import logging
import re
import time

import requests

logger = logging.getLogger(__name__)


class RawPost:
    def __init__(self, body: str, created_at: str):
        self.body = body
        self.created_at = created_at


class RedditSource:
    name = "reddit"

    def fetch(self, symbol: str, limit: int = 100) -> list[RawPost]:
        """Fetch recent r/wallstreetbets posts mentioning the ticker symbol."""
        url = "https://www.reddit.com/r/wallstreetbets/new.json"
        headers = {"User-Agent": "StockTradingBot/1.0 (research)"}
        params = {"limit": 100}

        # Build symbol match patterns: $SYMBOL or bare SYMBOL as word boundary
        patterns = [rf'\${re.escape(symbol.upper())}\b', rf'\b{re.escape(symbol.upper())}\b']

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            posts_data = data.get("data", {}).get("children", [])
            matched = []
            for child in posts_data:
                post = child.get("data", {})
                title = post.get("title", "")
                selftext = post.get("selftext", "")
                text_full = title + " " + selftext[:500]

                # Check if post mentions the ticker
                for pat in patterns:
                    if re.search(pat, text_full, re.IGNORECASE):
                        created_at = str(post.get("created_utc", ""))
                        ts = ""
                        try:
                            ts = str(created_at)
                        except Exception:
                            pass
                        matched.append(RawPost(body=text_full, created_at=ts))
                        break  # one match per post is enough
            return matched
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                logger.warning("Reddit rate limited for %s", symbol)
                return []
            logger.warning("Reddit HTTP error for %s: %s", symbol, e)
            return []
        except Exception as e:
            logger.warning("Reddit fetch failed for %s: %s", symbol, e)
            return []
        finally:
            time.sleep(1.0)  # polite rate limiting between requests
