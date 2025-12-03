"""
Tests for the multimodal highlight generation system.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimodal_highlight_gen import HighlightConfig, MultimodalHighlightPipeline
from multimodal_highlight_gen.analyzers import AudioAnalyzer, VideoAnalyzer, TextAnalyzer
from multimodal_highlight_gen.analyzers.text_analyzer import CommentaryEntry
from multimodal_highlight_gen.fusion import MultimodalFusionEngine
from multimodal_highlight_gen.utils.segment_utils import (
    create_highlight_segments,
    merge_overlapping_segments,
    HighlightSegment,
)


class TestAudioAnalyzer(unittest.TestCase):
    """Tests for AudioAnalyzer."""

    def test_initialization(self):
        """Test analyzer initialization with default parameters."""
        analyzer = AudioAnalyzer()
        self.assertEqual(analyzer.sample_rate, 22050)
        self.assertEqual(analyzer.crowd_roar_threshold, 0.7)

    def test_initialization_custom(self):
        """Test analyzer initialization with custom parameters."""
        analyzer = AudioAnalyzer(
            sample_rate=44100,
            crowd_roar_threshold=0.8,
        )
        self.assertEqual(analyzer.sample_rate, 44100)
        self.assertEqual(analyzer.crowd_roar_threshold, 0.8)

    def test_normalize(self):
        """Test normalization function."""
        analyzer = AudioAnalyzer()
        data = np.array([0, 5, 10])
        normalized = analyzer._normalize(data)
        self.assertAlmostEqual(normalized[0], 0.0)
        self.assertAlmostEqual(normalized[1], 0.5)
        self.assertAlmostEqual(normalized[2], 1.0)

    def test_normalize_constant(self):
        """Test normalization with constant values."""
        analyzer = AudioAnalyzer()
        data = np.array([5, 5, 5])
        normalized = analyzer._normalize(data)
        np.testing.assert_array_equal(normalized, np.zeros(3))


class TestVideoAnalyzer(unittest.TestCase):
    """Tests for VideoAnalyzer."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = VideoAnalyzer()
        self.assertEqual(analyzer.target_fps, 5)
        self.assertEqual(analyzer.emotion_threshold, 0.6)

    def test_compute_intensity_empty(self):
        """Test intensity computation with empty list."""
        analyzer = VideoAnalyzer()
        intensity = analyzer._compute_intensity([])
        self.assertEqual(intensity, 0.0)

    def test_compute_intensity(self):
        """Test intensity computation with emotions."""
        analyzer = VideoAnalyzer()
        emotions_list = [
            {"happy": 0.5, "surprise": 0.3, "neutral": 0.2},
            {"happy": 0.7, "surprise": 0.2, "neutral": 0.1},
        ]
        intensity = analyzer._compute_intensity(emotions_list)
        self.assertGreater(intensity, 0.0)


class TestTextAnalyzer(unittest.TestCase):
    """Tests for TextAnalyzer."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = TextAnalyzer()
        self.assertIsNotNone(analyzer.excitement_keywords)
        self.assertEqual(analyzer.explosion_threshold, 0.8)

    def test_find_keywords(self):
        """Test keyword detection."""
        analyzer = TextAnalyzer()
        keywords = analyzer._find_keywords("GOAL! What an incredible save!")
        self.assertIn("goal", keywords)
        self.assertIn("incredible", keywords)
        self.assertIn("save", keywords)

    def test_find_keywords_empty(self):
        """Test keyword detection with no keywords."""
        analyzer = TextAnalyzer()
        keywords = analyzer._find_keywords("Normal boring text")
        self.assertEqual(keywords, [])

    def test_compute_excitement_score_high(self):
        """Test excitement scoring for high-excitement text."""
        analyzer = TextAnalyzer()
        score = analyzer._compute_excitement_score(
            "GOAL!!! INCREDIBLE!!! UNBELIEVABLE!!!"
        )
        self.assertGreater(score, 0.5)

    def test_compute_excitement_score_low(self):
        """Test excitement scoring for low-excitement text."""
        analyzer = TextAnalyzer()
        score = analyzer._compute_excitement_score(
            "The ball is passed around"
        )
        self.assertLess(score, 0.3)

    def test_analyze_text(self):
        """Test full text analysis."""
        analyzer = TextAnalyzer()
        result = analyzer.analyze_text("GOAL! Amazing strike!")
        self.assertIn("excitement_score", result)
        self.assertIn("sentiment_score", result)
        self.assertIn("keywords_found", result)
        self.assertIn("is_explosive", result)


class TestMultimodalFusion(unittest.TestCase):
    """Tests for MultimodalFusionEngine."""

    def test_initialization(self):
        """Test fusion engine initialization."""
        fusion = MultimodalFusionEngine()
        # Weights should be normalized
        total = fusion.audio_weight + fusion.video_weight + fusion.text_weight
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_weighted_average_fusion(self):
        """Test weighted average fusion."""
        fusion = MultimodalFusionEngine(
            audio_weight=0.5,
            video_weight=0.3,
            text_weight=0.2,
        )
        audio = np.array([1.0, 0.5, 0.0])
        video = np.array([0.0, 0.5, 1.0])
        text = np.array([0.5, 0.5, 0.5])

        fused = fusion._weighted_average_fusion(audio, video, text)
        self.assertEqual(len(fused), 3)
        # First element should be weighted toward audio
        self.assertGreater(fused[0], fused[2])

    def test_fuse(self):
        """Test full fusion process."""
        fusion = MultimodalFusionEngine()
        timestamps = np.arange(0, 10, 0.5)
        scores = np.random.rand(len(timestamps))

        result = fusion.fuse(
            audio_timestamps=timestamps,
            audio_scores=scores,
            video_timestamps=timestamps,
            video_scores=scores * 0.8,
            text_timestamps=timestamps,
            text_scores=scores * 0.6,
        )

        self.assertEqual(len(result.fused_scores), len(result.timestamps))
        self.assertEqual(len(result.audio_scores), len(result.timestamps))
        self.assertTrue(all(0 <= s <= 1 for s in result.fused_scores))


class TestSegmentUtils(unittest.TestCase):
    """Tests for segment utility functions."""

    def test_create_highlight_segments(self):
        """Test highlight segment creation."""
        timestamps = np.arange(0, 60, 0.5)
        scores = np.zeros_like(timestamps)
        scores[20:30] = 0.8  # Peak at 10-15s

        segments = create_highlight_segments(
            timestamps=timestamps,
            scores=scores,
            min_score=0.6,
            min_duration=1.0,
        )

        self.assertGreater(len(segments), 0)
        self.assertGreater(segments[0].score, 0.6)

    def test_merge_overlapping_segments(self):
        """Test segment merging."""
        segments = [
            HighlightSegment(
                start_time=0, end_time=10, score=0.8, peak_time=5,
                audio_contribution=0.3, video_contribution=0.3,
                text_contribution=0.3
            ),
            HighlightSegment(
                start_time=12, end_time=20, score=0.7, peak_time=15,
                audio_contribution=0.3, video_contribution=0.3,
                text_contribution=0.3
            ),
        ]

        merged = merge_overlapping_segments(segments, merge_threshold=5.0)
        # Gap of 2s should trigger merge with threshold 5s
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].start_time, 0)
        self.assertEqual(merged[0].end_time, 20)


class TestHighlightConfig(unittest.TestCase):
    """Tests for HighlightConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = HighlightConfig()
        self.assertEqual(config.audio.sample_rate, 22050)
        self.assertEqual(config.fusion.audio_weight, 0.35)
        self.assertEqual(config.min_highlight_duration, 3.0)

    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = HighlightConfig()
        config_dict = config.to_dict()
        self.assertIn("audio", config_dict)
        self.assertIn("video", config_dict)
        self.assertIn("text", config_dict)
        self.assertIn("fusion", config_dict)

    def test_config_from_dict(self):
        """Test configuration deserialization."""
        config_dict = {
            "audio": {"sample_rate": 44100},
            "fusion": {"audio_weight": 0.5},
            "min_highlight_duration": 5.0,
        }
        config = HighlightConfig.from_dict(config_dict)
        self.assertEqual(config.audio.sample_rate, 44100)
        self.assertEqual(config.fusion.audio_weight, 0.5)
        self.assertEqual(config.min_highlight_duration, 5.0)


class TestMultimodalHighlightPipeline(unittest.TestCase):
    """Tests for the main pipeline."""

    def test_initialization(self):
        """Test pipeline initialization."""
        pipeline = MultimodalHighlightPipeline()
        self.assertIsNotNone(pipeline.audio_analyzer)
        self.assertIsNotNone(pipeline.video_analyzer)
        self.assertIsNotNone(pipeline.text_analyzer)
        self.assertIsNotNone(pipeline.fusion_engine)

    def test_initialization_custom_config(self):
        """Test pipeline initialization with custom config."""
        config = HighlightConfig()
        config.fusion.audio_weight = 0.6
        config.fusion.video_weight = 0.2
        config.fusion.text_weight = 0.2
        pipeline = MultimodalHighlightPipeline(config)
        # Weight should be normalized - total should be 1.0
        total = (
            pipeline.fusion_engine.audio_weight
            + pipeline.fusion_engine.video_weight
            + pipeline.fusion_engine.text_weight
        )
        self.assertAlmostEqual(total, 1.0, places=5)
        # Audio weight should be highest
        self.assertGreater(
            pipeline.fusion_engine.audio_weight,
            pipeline.fusion_engine.video_weight
        )


if __name__ == "__main__":
    unittest.main()
