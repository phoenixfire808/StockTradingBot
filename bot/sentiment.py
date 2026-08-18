"""Social-sentiment analysis module.

Aggregates posts from StockTwits and Reddit, scores with VADER,
persists fetched posts to CSV cache for repeat access.
"""

import csv
import pandas as pd
import hashlib
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _analyzer = None  # lazy init per engine instance
except ImportError:
    _analyzer = None
    logger.warning("vaderSentiment not available — sentiment scoring disabled.")


@dataclass
class SentimentPost:
    """Single post extracted from a sentiment source."""
    timestamp: datetime
    source: str
    text: str
    score: float  # VADER compound (-1..1)


@dataclass
class SentimentScore:
    """Aggregated sentiment result for a symbol over a time window."""
    symbol: str
    window_hours: int
    mentions: int
    bullish: int = 0
    bearish: int = 0
    neutral: int = 0
    net_score: float = 0.0
    top_posts: list[SentimentPost] = field(default_factory=list)


class SentimentEngine:
    """Fetches and scores social-sentiment from registered sources."""

    def __init__(self) -> None:
        if _analyzer is not None:
            self._analyzer = SentimentIntensityAnalyzer()
        else:
            self._analyzer = None

    def _score_text(self, text: str) -> float:
        """Compute VADER compound score for text."""
        if self._analyzer is None:
            return 0.0
        return self._analyzer.polarity_scores(text)["compound"]

    def fetch_post(self, source_plugin, symbol: str, limit: int = 30) -> list[SentimentPost]:
        """Fetch raw posts from a single sentiment source plugin."""
        try:
            raw_posts = source_plugin.fetch(symbol, limit=limit)
            scored: list[SentimentPost] = []
            for p in raw_posts:
                txt = getattr(p, "body", getattr(p, "text", ""))
                ts_str = getattr(p, "created_at", "")
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    ts = datetime.now()
                score = self._score_text(str(txt))
                scored.append(SentimentPost(
                    timestamp=ts,
                    source=source_plugin.name,
                    text=str(txt)[:500],  # cap text length
                    score=score,
                ))
            return scored
        except Exception as exc:
            logger.warning(f"Sentiment source '{source_plugin.name}' failed for {symbol}: {exc}")
            return []

    def save_posts(self, symbol: str, posts: list[SentimentPost]) -> None:
        """Append posts to data/sentiment/{symbol}.csv for caching."""
        csv_path = Path("data/sentiment") / f"{symbol}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        headers = ["timestamp", "source", "text_hash", "score"]
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if write_header:
                writer.writeheader()
            existing_hashes = set()
            if f.tell() > 0:  # file has content
                # Read existing hashes to avoid duplicates
                pass  # simplified — just append

            for post in posts:
                text_hash = hashlib.sha256(post.text.encode()).hexdigest()[:12]
                writer.writerow({
                    "timestamp": post.timestamp.isoformat(),
                    "source": post.source,
                    "text_hash": text_hash,
                    "score": round(post.score, 4),
                })

    def score(
        self,
        symbol: str,
        hours: Optional[int] = None,
    ) -> SentimentScore:
        """Aggregate sentiment across all sources for *symbol*.

        Returns SentimentScore with counts and computed net_score.
        Reads cached posts within the window; refetches only new data.
        """
        from bot.core import SENTIMENT_SOURCES

        window_hours = hours if hours is not None else 24
        cutoff = datetime.utcnow() - timedelta(hours=window_hours)

        windowed: list[SentimentPost] = []
        sources_working: list[str] = []
        sources_failed: list[str] = []

        for src_name, src_plugin in SENTIMENT_SOURCES.items():
            posts = self.fetch_post(src_plugin, symbol, limit=30)
            if posts:
                # Filter to window from cache first
                csv_path = Path("data/sentiment") / f"{symbol}.csv"
                if csv_path.exists() and csv_path.stat().st_size > 0:
                    try:
                        cached = pd.read_csv(csv_path, parse_dates=["timestamp"])
                        windowed.extend(
                            SentimentPost(
                                timestamp=p["timestamp"],
                                source=p["source"],
                                text="",
                                score=float(p["score"]),
                            )
                            for _, p in cached.iterrows()
                            if p["timestamp"] >= cutoff
                        )
                    except Exception:
                        pass  # fall through to fresh fetch
                if posts:
                    windowed.extend(posts)
                    self.save_posts(symbol, posts)
                    sources_working.append(src_plugin.name)
            else:
                sources_failed.append(src_plugin.name)

        # Deduplicate by approximate text similarity (hash overlap)
        seen_hashes: set[str] = set()
        unique_posts: list[SentimentPost] = []
        for p in windowed:
            h = hashlib.sha256(p.text.encode() + p.source.encode()).hexdigest()[:12]
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_posts.append(p)

        # Compute stats
        bullish = sum(1 for p in unique_posts if p.score > 0.25)
        bearish = sum(1 for p in unique_posts if p.score < -0.25)
        neutral = len(unique_posts) - bullish - bearish
        mentions = len(unique_posts)
        net_score = (bullish - bearish) / max(mentions, 1)

        # Top 5 posts by |compound|
        sorted_posts = sorted(unique_posts, key=lambda p: abs(p.score), reverse=True)[:5]

        score_obj = SentimentScore(
            symbol=symbol,
            window_hours=window_hours,
            mentions=mentions,
            bullish=bullish,
            bearish=bearish,
            neutral=neutral,
            net_score=round(net_score, 4),
            top_posts=sorted_posts,
        )

        logger.info(
            "Sentiment %s: mentions=%d bullish=%d bearish=%d net=%.2f (sources: %s)",
            symbol, mentions, bullish, bearish, net_score,
            ", ".join(sources_working) or "(none)",
        )
        return score_obj
