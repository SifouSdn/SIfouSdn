"""
Configuration module for Multimodal Highlight Generation system.

Contains all configurable parameters for the highlight generation pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AudioConfig:
    """Configuration for audio analysis."""

    sample_rate: int = 22050
    hop_length: int = 512
    n_mels: int = 128
    crowd_roar_threshold: float = 0.7
    min_roar_duration: float = 1.0  # seconds
    energy_percentile: float = 90.0
    window_size: float = 2.0  # seconds for analysis window


@dataclass
class VideoConfig:
    """Configuration for video analysis."""

    target_fps: int = 5  # Frames per second to analyze
    face_detection_confidence: float = 0.5
    emotion_threshold: float = 0.6
    intense_emotions: List[str] = field(
        default_factory=lambda: ["happy", "surprise", "anger", "fear"]
    )
    batch_size: int = 16
    resize_width: int = 640
    resize_height: int = 480


@dataclass
class TextConfig:
    """Configuration for text/commentary analysis."""

    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    excitement_keywords: List[str] = field(
        default_factory=lambda: [
            "goal",
            "score",
            "amazing",
            "incredible",
            "unbelievable",
            "fantastic",
            "brilliant",
            "spectacular",
            "save",
            "miss",
            "chance",
            "shot",
            "header",
            "penalty",
            "red card",
            "yellow card",
            "foul",
            "offside",
            "corner",
            "free kick",
            "injury",
            "substitution",
        ]
    )
    explosion_threshold: float = 0.8
    max_sequence_length: int = 512


@dataclass
class FusionConfig:
    """Configuration for multimodal fusion."""

    audio_weight: float = 0.35
    video_weight: float = 0.30
    text_weight: float = 0.35
    fusion_method: str = "weighted_average"  # Options: weighted_average, attention, late_fusion
    temporal_smoothing_window: float = 3.0  # seconds
    min_highlight_score: float = 0.65


@dataclass
class HighlightConfig:
    """Main configuration for highlight generation."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    text: TextConfig = field(default_factory=TextConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    # Highlight generation parameters
    min_highlight_duration: float = 3.0  # seconds
    max_highlight_duration: float = 30.0  # seconds
    context_before: float = 2.0  # seconds before peak
    context_after: float = 5.0  # seconds after peak
    merge_threshold: float = 5.0  # seconds - merge highlights closer than this
    max_highlights: Optional[int] = None  # None means no limit
    output_format: str = "mp4"

    @classmethod
    def from_dict(cls, config_dict: dict) -> "HighlightConfig":
        """Create configuration from dictionary."""
        audio_config = AudioConfig(**config_dict.get("audio", {}))
        video_config = VideoConfig(**config_dict.get("video", {}))
        text_config = TextConfig(**config_dict.get("text", {}))
        fusion_config = FusionConfig(**config_dict.get("fusion", {}))

        return cls(
            audio=audio_config,
            video=video_config,
            text=text_config,
            fusion=fusion_config,
            min_highlight_duration=config_dict.get("min_highlight_duration", 3.0),
            max_highlight_duration=config_dict.get("max_highlight_duration", 30.0),
            context_before=config_dict.get("context_before", 2.0),
            context_after=config_dict.get("context_after", 5.0),
            merge_threshold=config_dict.get("merge_threshold", 5.0),
            max_highlights=config_dict.get("max_highlights"),
            output_format=config_dict.get("output_format", "mp4"),
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "audio": {
                "sample_rate": self.audio.sample_rate,
                "hop_length": self.audio.hop_length,
                "n_mels": self.audio.n_mels,
                "crowd_roar_threshold": self.audio.crowd_roar_threshold,
                "min_roar_duration": self.audio.min_roar_duration,
                "energy_percentile": self.audio.energy_percentile,
                "window_size": self.audio.window_size,
            },
            "video": {
                "target_fps": self.video.target_fps,
                "face_detection_confidence": self.video.face_detection_confidence,
                "emotion_threshold": self.video.emotion_threshold,
                "intense_emotions": self.video.intense_emotions,
                "batch_size": self.video.batch_size,
                "resize_width": self.video.resize_width,
                "resize_height": self.video.resize_height,
            },
            "text": {
                "sentiment_model": self.text.sentiment_model,
                "excitement_keywords": self.text.excitement_keywords,
                "explosion_threshold": self.text.explosion_threshold,
                "max_sequence_length": self.text.max_sequence_length,
            },
            "fusion": {
                "audio_weight": self.fusion.audio_weight,
                "video_weight": self.fusion.video_weight,
                "text_weight": self.fusion.text_weight,
                "fusion_method": self.fusion.fusion_method,
                "temporal_smoothing_window": self.fusion.temporal_smoothing_window,
                "min_highlight_score": self.fusion.min_highlight_score,
            },
            "min_highlight_duration": self.min_highlight_duration,
            "max_highlight_duration": self.max_highlight_duration,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "merge_threshold": self.merge_threshold,
            "max_highlights": self.max_highlights,
            "output_format": self.output_format,
        }
