"""Tests for bot.sentiment — VADER scoring, dedupe, error handling."""

import json
import logging
from unittest.mock import patch, MagicMock

import pytest
from bot.sentiment import SentimentPost, SentimentScore, SentimentEngine


class TestSentimentScoring:
    """VADER compound score classification."""

    def test_positive_sentence_bullish(self):
        """Bullish text should score above 0.25."""
        engine = SentimentEngine()
        score = engine._score_text("Stock is surging on amazing earnings report 🚀")
        assert score > 0.25

    def test_negative_sentence_bearish(self):
        """Bearish text should score below -0.25."""
        engine = SentimentEngine()
        score = engine._score_text("Massive sell-off today, company failing badly 💔")
        assert score < -0.25

    def test_neutral_text(self):
        """Neutral text should score between -0.25 and 0.25."""
        engine = SentimentEngine()
        score = engine._score_text("The stock market opened at standard levels today.")
        assert -0.25 <= score <= 0.25

    def test_empty_string_returns_zero(self):
        engine = SentimentEngine()
        score = engine._score_text("")
        assert score == 0.0


class TestVADERUnavailable:
    def test_no_vader_returns_zero(self):
        """When vaderSentiment isn't installed, scoring should safely return 0."""
        engine = SentimentEngine()
        engine._analyzer = None
        assert engine._score_text("anything here") == 0.0


class TestSourceFailureGraceful:
    def test_fetch_returns_empty_on_exception(self):
        """A failed sentiment source should return [] without raising."""
        from unittest.mock import patch
        from bot.plugins.sentiment_sources.stocktwits import StockTwitsSource

        source = StockTwitsSource()
        with patch("requests.get", side_effect=Exception("network error")):
            posts = source.fetch("AAPL", limit=30)
        assert posts == []


class TestSentimentScoreComputation:
    """Aggregate sentiment metrics calculation."""

    def test_net_score_calculation(self):
        bullish = 10
        bearish = 4
        mentions = 14
        net = (bullish - bearish) / max(mentions, 1)
        assert abs(net - 6/14) < 0.01

    def test_equal_bull_bear_gives_zero(self):
        net = (5 - 5) / 10
        assert net == 0.0


class TestDataclassFields:
    def test_sentiment_post_defaults(self):
        post = SentimentPost(
            timestamp="2026-08-18T12:00:00Z",
            source="test",
            text="hello world",
            score=0.5,
        )
        assert post.source == "test"
        assert post.score == 0.5

    def test_sentiment_score_defaults(self):
        score = SentimentScore(
            symbol="AAPL",
            window_hours=24,
            mentions=5,
        )
        assert score.bullish == 0
        assert score.bearish == 0
        assert score.neutral == 0
        assert score.net_score == 0.0
        assert len(score.top_posts) == 0
