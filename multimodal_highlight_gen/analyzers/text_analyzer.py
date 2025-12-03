"""
Text Analyzer for Live Commentary Sentiment Analysis.

This module analyzes text from live commentary to detect
explosive reactions that indicate highlight-worthy moments.
"""

import re
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TextEvent:
    """Represents a detected text event (explosive commentary)."""

    start_time: float  # seconds
    end_time: float  # seconds
    text: str
    sentiment_score: float  # -1.0 to 1.0 (negative to positive)
    excitement_score: float  # 0.0 to 1.0
    keywords_found: List[str]
    confidence: float


@dataclass
class CommentaryEntry:
    """A single commentary entry with timestamp."""

    timestamp: float  # seconds
    text: str
    speaker: Optional[str] = None


class TextAnalyzer:
    """
    Analyzes live commentary text to detect explosive reactions.

    Uses sentiment analysis and keyword detection to identify
    moments of high excitement in sports commentary.
    """

    DEFAULT_EXCITEMENT_KEYWORDS = [
        # Goals and scoring
        "goal",
        "score",
        "scores",
        "scored",
        "net",
        "back of the net",
        # Excitement expressions
        "amazing",
        "incredible",
        "unbelievable",
        "fantastic",
        "brilliant",
        "spectacular",
        "magnificent",
        "stunning",
        "outstanding",
        "phenomenal",
        "sensational",
        "extraordinary",
        "wonderful",
        "superb",
        # Actions
        "save",
        "saved",
        "miss",
        "missed",
        "chance",
        "shot",
        "header",
        "volley",
        "strike",
        "finish",
        # Game events
        "penalty",
        "red card",
        "yellow card",
        "foul",
        "offside",
        "corner",
        "free kick",
        "injury",
        "substitution",
        # Intensity markers
        "what a",
        "oh my",
        "can you believe",
        "history",
        "record",
        "comeback",
        "equalizer",
        "winner",
        "hat trick",
        "hat-trick",
    ]

    INTENSITY_MARKERS = [
        "!",
        "!!",
        "!!!",
        "?!",
        "...",
        "GOAL",
        "YES",
        "WOW",
        "OH",
    ]

    def __init__(
        self,
        sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
        excitement_keywords: Optional[List[str]] = None,
        explosion_threshold: float = 0.8,
        max_sequence_length: int = 512,
    ):
        """
        Initialize the text analyzer.

        Args:
            sentiment_model: Name of the sentiment analysis model
            excitement_keywords: List of keywords indicating excitement
            explosion_threshold: Threshold for explosive reaction detection
            max_sequence_length: Maximum sequence length for transformer
        """
        self.sentiment_model = sentiment_model
        self.excitement_keywords = (
            excitement_keywords or self.DEFAULT_EXCITEMENT_KEYWORDS
        )
        self.explosion_threshold = explosion_threshold
        self.max_sequence_length = max_sequence_length

        self._sentiment_pipeline = None
        self._transformer_available = False

    def _ensure_dependencies(self):
        """Lazy load heavy dependencies."""
        if self._sentiment_pipeline is not None:
            return

        try:
            from transformers import pipeline

            self._sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.sentiment_model,
            )
            self._transformer_available = True
            logger.info(f"Loaded sentiment model: {self.sentiment_model}")
        except ImportError:
            logger.warning(
                "transformers not installed. Using rule-based sentiment. "
                "Install with: pip install transformers"
            )
            self._transformer_available = False
        except Exception as e:
            logger.warning(f"Could not load sentiment model: {e}")
            self._transformer_available = False

    def analyze(
        self,
        commentary: List[CommentaryEntry],
    ) -> List[TextEvent]:
        """
        Analyze commentary for explosive reaction events.

        Args:
            commentary: List of CommentaryEntry objects with timestamps

        Returns:
            List of detected TextEvent objects
        """
        self._ensure_dependencies()

        if not commentary:
            return []

        events = []

        for entry in commentary:
            excitement_score = self._compute_excitement_score(entry.text)
            sentiment_score = self._compute_sentiment_score(entry.text)
            keywords = self._find_keywords(entry.text)

            # Detect if this is an explosive reaction
            is_explosive = (
                excitement_score >= self.explosion_threshold
                or len(keywords) >= 2
            )

            if is_explosive:
                events.append(
                    TextEvent(
                        start_time=entry.timestamp,
                        end_time=entry.timestamp + 5.0,  # Assume 5s window
                        text=entry.text,
                        sentiment_score=sentiment_score,
                        excitement_score=excitement_score,
                        keywords_found=keywords,
                        confidence=excitement_score,
                    )
                )

        logger.info(f"Detected {len(events)} text events")
        return events

    def get_excitement_timeline(
        self,
        commentary: List[CommentaryEntry],
        time_resolution: float = 0.5,
        max_time: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get excitement scores over time from commentary.

        Args:
            commentary: List of CommentaryEntry objects
            time_resolution: Time resolution in seconds
            max_time: Maximum time (defaults to last commentary timestamp)

        Returns:
            Tuple of (timestamps, excitement_scores)
        """
        if not commentary:
            return np.array([]), np.array([])

        if max_time is None:
            max_time = max(c.timestamp for c in commentary) + 10.0

        timestamps = np.arange(0, max_time, time_resolution)
        excitement_scores = np.zeros_like(timestamps)

        # Compute excitement for each commentary entry
        for entry in commentary:
            score = self._compute_excitement_score(entry.text)

            # Apply score to relevant time window (decay over time)
            for i, t in enumerate(timestamps):
                if entry.timestamp <= t < entry.timestamp + 10.0:
                    decay = 1.0 - (t - entry.timestamp) / 10.0
                    excitement_scores[i] = max(
                        excitement_scores[i], score * decay
                    )

        return timestamps, excitement_scores

    def analyze_text(self, text: str) -> Dict[str, any]:
        """
        Analyze a single text for excitement and sentiment.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with analysis results
        """
        self._ensure_dependencies()

        excitement_score = self._compute_excitement_score(text)
        sentiment_score = self._compute_sentiment_score(text)
        keywords = self._find_keywords(text)

        return {
            "text": text,
            "excitement_score": excitement_score,
            "sentiment_score": sentiment_score,
            "keywords_found": keywords,
            "is_explosive": excitement_score >= self.explosion_threshold,
        }

    def _compute_excitement_score(self, text: str) -> float:
        """Compute excitement score from text."""
        if not text:
            return 0.0

        score = 0.0

        # Keyword presence
        keywords = self._find_keywords(text)
        keyword_score = min(1.0, len(keywords) * 0.2)

        # Intensity markers (exclamation points, caps)
        text_upper = text.upper()
        caps_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
        exclamation_count = text.count("!")

        intensity_score = min(1.0, caps_ratio * 2 + exclamation_count * 0.15)

        # Text length (longer excited commentary)
        length_score = min(1.0, len(text.split()) / 20.0) * 0.3

        # Check for intensity markers
        marker_score = 0.0
        for marker in self.INTENSITY_MARKERS:
            if marker in text_upper:
                marker_score = max(marker_score, 0.3)

        # Combine scores
        score = (
            0.4 * keyword_score
            + 0.3 * intensity_score
            + 0.15 * length_score
            + 0.15 * marker_score
        )

        return min(1.0, score)

    def _compute_sentiment_score(self, text: str) -> float:
        """Compute sentiment score (-1 to 1) from text."""
        if not text:
            return 0.0

        if self._transformer_available and self._sentiment_pipeline:
            try:
                result = self._sentiment_pipeline(
                    text[: self.max_sequence_length]
                )[0]
                label = result["label"]
                confidence = result["score"]

                if label == "POSITIVE":
                    return confidence
                else:
                    return -confidence
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
                return self._rule_based_sentiment(text)
        else:
            return self._rule_based_sentiment(text)

    def _rule_based_sentiment(self, text: str) -> float:
        """Simple rule-based sentiment analysis."""
        positive_words = {
            "goal",
            "amazing",
            "incredible",
            "fantastic",
            "brilliant",
            "wonderful",
            "great",
            "excellent",
            "superb",
            "beautiful",
            "perfect",
            "win",
            "winner",
            "victory",
            "triumph",
            "save",
            "saved",
        }

        negative_words = {
            "miss",
            "missed",
            "foul",
            "injury",
            "injured",
            "terrible",
            "awful",
            "poor",
            "bad",
            "loss",
            "lost",
            "fail",
            "failed",
            "mistake",
            "error",
        }

        text_lower = text.lower()
        words = set(re.findall(r"\b\w+\b", text_lower))

        positive_count = len(words & positive_words)
        negative_count = len(words & negative_words)

        if positive_count + negative_count == 0:
            return 0.0

        return (positive_count - negative_count) / (positive_count + negative_count)

    def _find_keywords(self, text: str) -> List[str]:
        """Find excitement keywords in text."""
        text_lower = text.lower()
        found = []

        for keyword in self.excitement_keywords:
            if keyword.lower() in text_lower:
                found.append(keyword)

        return found
