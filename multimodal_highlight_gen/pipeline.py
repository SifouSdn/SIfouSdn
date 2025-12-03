"""
Multimodal Highlight Generation Pipeline.

This module orchestrates the complete highlight generation process,
combining audio, video, and text analysis with multimodal fusion.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import logging
import os
import json

from .config import HighlightConfig
from .analyzers import AudioAnalyzer, VideoAnalyzer, TextAnalyzer
from .analyzers.text_analyzer import CommentaryEntry
from .fusion import MultimodalFusionEngine
from .utils.segment_utils import (
    HighlightSegment,
    create_highlight_segments,
    merge_overlapping_segments,
    select_top_segments,
)
from .utils.video_utils import extract_audio_from_video, get_video_duration

logger = logging.getLogger(__name__)


@dataclass
class HighlightResult:
    """Result of highlight generation."""

    segments: List[HighlightSegment]
    metadata: Dict[str, Any] = field(default_factory=dict)
    fusion_scores: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "segments": [
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "duration": seg.duration,
                    "score": seg.score,
                    "peak_time": seg.peak_time,
                    "audio_contribution": seg.audio_contribution,
                    "video_contribution": seg.video_contribution,
                    "text_contribution": seg.text_contribution,
                    "label": seg.label,
                }
                for seg in self.segments
            ],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class MultimodalHighlightPipeline:
    """
    Main pipeline for multimodal highlight generation.

    Coordinates audio, video, and text analysis to automatically
    detect and extract highlight moments from sports videos.
    """

    def __init__(self, config: Optional[HighlightConfig] = None):
        """
        Initialize the highlight generation pipeline.

        Args:
            config: Configuration object (uses defaults if not provided)
        """
        self.config = config or HighlightConfig()

        # Initialize analyzers
        self.audio_analyzer = AudioAnalyzer(
            sample_rate=self.config.audio.sample_rate,
            hop_length=self.config.audio.hop_length,
            n_mels=self.config.audio.n_mels,
            crowd_roar_threshold=self.config.audio.crowd_roar_threshold,
            min_roar_duration=self.config.audio.min_roar_duration,
            energy_percentile=self.config.audio.energy_percentile,
            window_size=self.config.audio.window_size,
        )

        self.video_analyzer = VideoAnalyzer(
            target_fps=self.config.video.target_fps,
            face_detection_confidence=self.config.video.face_detection_confidence,
            emotion_threshold=self.config.video.emotion_threshold,
            intense_emotions=self.config.video.intense_emotions,
            batch_size=self.config.video.batch_size,
            resize_width=self.config.video.resize_width,
            resize_height=self.config.video.resize_height,
        )

        self.text_analyzer = TextAnalyzer(
            sentiment_model=self.config.text.sentiment_model,
            excitement_keywords=self.config.text.excitement_keywords,
            explosion_threshold=self.config.text.explosion_threshold,
            max_sequence_length=self.config.text.max_sequence_length,
        )

        # Initialize fusion engine
        self.fusion_engine = MultimodalFusionEngine(
            audio_weight=self.config.fusion.audio_weight,
            video_weight=self.config.fusion.video_weight,
            text_weight=self.config.fusion.text_weight,
            fusion_method=self.config.fusion.fusion_method,
            temporal_smoothing_window=self.config.fusion.temporal_smoothing_window,
            min_highlight_score=self.config.fusion.min_highlight_score,
        )

        logger.info("Initialized MultimodalHighlightPipeline")

    def generate_highlights(
        self,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        commentary: Optional[List[CommentaryEntry]] = None,
        audio_data: Optional[Tuple[np.ndarray, int]] = None,
        video_frames: Optional[Tuple[List[np.ndarray], List[float]]] = None,
    ) -> HighlightResult:
        """
        Generate highlights from multimodal inputs.

        Args:
            video_path: Path to video file
            audio_path: Path to audio file (uses video audio if not provided)
            commentary: List of CommentaryEntry objects for text analysis
            audio_data: Pre-loaded audio as (data, sample_rate) tuple
            video_frames: Pre-loaded frames as (frames, timestamps) tuple

        Returns:
            HighlightResult with detected highlights
        """
        logger.info("Starting highlight generation...")

        # Validate inputs
        if video_path is None and audio_data is None and video_frames is None:
            raise ValueError(
                "At least one of video_path, audio_data, or video_frames must be provided"
            )

        # Get video duration for metadata
        duration = 0.0
        if video_path:
            try:
                duration = get_video_duration(video_path)
            except Exception as e:
                logger.warning(f"Could not get video duration: {e}")

        # Analyze each modality
        audio_timestamps, audio_scores = self._analyze_audio(
            video_path, audio_path, audio_data
        )

        video_timestamps, video_scores = self._analyze_video(
            video_path, video_frames
        )

        text_timestamps, text_scores = self._analyze_text(
            commentary, duration or max(
                audio_timestamps[-1] if len(audio_timestamps) > 0 else 0,
                video_timestamps[-1] if len(video_timestamps) > 0 else 0,
            )
        )

        # Fuse modalities
        fusion_result = self.fusion_engine.fuse(
            audio_timestamps=audio_timestamps,
            audio_scores=audio_scores,
            video_timestamps=video_timestamps,
            video_scores=video_scores,
            text_timestamps=text_timestamps,
            text_scores=text_scores,
        )

        # Create highlight segments
        segments = create_highlight_segments(
            timestamps=fusion_result.timestamps,
            scores=fusion_result.fused_scores,
            min_score=self.config.fusion.min_highlight_score,
            min_duration=self.config.min_highlight_duration,
            max_duration=self.config.max_highlight_duration,
            context_before=self.config.context_before,
            context_after=self.config.context_after,
            audio_scores=fusion_result.audio_scores,
            video_scores=fusion_result.video_scores,
            text_scores=fusion_result.text_scores,
        )

        # Merge overlapping segments
        segments = merge_overlapping_segments(
            segments, self.config.merge_threshold
        )

        # Select top segments if limit specified
        if self.config.max_highlights:
            segments = select_top_segments(
                segments, max_segments=self.config.max_highlights
            )

        # Build metadata
        metadata = {
            "video_path": video_path,
            "duration": duration,
            "num_segments": len(segments),
            "total_highlight_duration": sum(s.duration for s in segments),
            "config": self.config.to_dict(),
        }

        logger.info(
            f"Generated {len(segments)} highlights "
            f"(total duration: {metadata['total_highlight_duration']:.2f}s)"
        )

        return HighlightResult(
            segments=segments,
            metadata=metadata,
            fusion_scores=fusion_result.fused_scores,
            timestamps=fusion_result.timestamps,
        )

    def _analyze_audio(
        self,
        video_path: Optional[str],
        audio_path: Optional[str],
        audio_data: Optional[Tuple[np.ndarray, int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Analyze audio modality."""
        logger.info("Analyzing audio...")

        try:
            if audio_data is not None:
                data, sr = audio_data
                return self.audio_analyzer.get_excitement_timeline(
                    audio_data=data, sr=sr
                )
            elif audio_path is not None:
                return self.audio_analyzer.get_excitement_timeline(
                    audio_path=audio_path
                )
            elif video_path is not None:
                return self.audio_analyzer.get_excitement_timeline(
                    audio_path=video_path
                )
            else:
                logger.warning("No audio source available")
                return np.array([]), np.array([])
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return np.array([]), np.array([])

    def _analyze_video(
        self,
        video_path: Optional[str],
        video_frames: Optional[Tuple[List[np.ndarray], List[float]]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Analyze video modality."""
        logger.info("Analyzing video...")

        try:
            if video_frames is not None:
                frames, timestamps = video_frames
                return self.video_analyzer.get_intensity_timeline(
                    frames=frames, timestamps=timestamps
                )
            elif video_path is not None:
                return self.video_analyzer.get_intensity_timeline(
                    video_path=video_path
                )
            else:
                logger.warning("No video source available")
                return np.array([]), np.array([])
        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            return np.array([]), np.array([])

    def _analyze_text(
        self,
        commentary: Optional[List[CommentaryEntry]],
        max_time: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Analyze text modality."""
        logger.info("Analyzing text...")

        if commentary is None or len(commentary) == 0:
            logger.warning("No commentary available for text analysis")
            return np.array([]), np.array([])

        try:
            return self.text_analyzer.get_excitement_timeline(
                commentary=commentary,
                max_time=max_time if max_time > 0 else None,
            )
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return np.array([]), np.array([])

    def analyze_single_modality(
        self,
        modality: str,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        commentary: Optional[List[CommentaryEntry]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyze a single modality.

        Args:
            modality: One of 'audio', 'video', 'text'
            video_path: Path to video file
            audio_path: Path to audio file
            commentary: Commentary entries for text analysis

        Returns:
            Tuple of (timestamps, scores)
        """
        if modality == "audio":
            return self._analyze_audio(video_path, audio_path, None)
        elif modality == "video":
            return self._analyze_video(video_path, None)
        elif modality == "text":
            max_time = 0.0
            if video_path:
                try:
                    max_time = get_video_duration(video_path)
                except Exception:
                    pass
            return self._analyze_text(commentary, max_time)
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def export_highlights(
        self,
        result: HighlightResult,
        video_path: str,
        output_dir: str,
        format: Optional[str] = None,
    ) -> List[str]:
        """
        Export highlight segments as separate video files.

        Args:
            result: HighlightResult from generate_highlights()
            video_path: Path to source video
            output_dir: Directory to save highlights
            format: Output format (defaults to config.output_format)

        Returns:
            List of output file paths
        """
        try:
            from moviepy.editor import VideoFileClip
        except ImportError:
            raise ImportError(
                "moviepy required for video export. "
                "Install with: pip install moviepy"
            )

        os.makedirs(output_dir, exist_ok=True)
        output_format = format or self.config.output_format

        output_paths = []
        video = VideoFileClip(video_path)

        for i, segment in enumerate(result.segments):
            output_path = os.path.join(
                output_dir,
                f"highlight_{i+1:03d}_{segment.start_time:.1f}s.{output_format}",
            )

            # Extract segment
            clip = video.subclip(segment.start_time, segment.end_time)
            clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                logger=None,
            )
            clip.close()

            output_paths.append(output_path)
            logger.info(f"Exported highlight {i+1}: {output_path}")

        video.close()

        return output_paths

    def get_summary(self, result: HighlightResult) -> str:
        """
        Get a human-readable summary of the highlight result.

        Args:
            result: HighlightResult from generate_highlights()

        Returns:
            Summary string
        """
        lines = [
            "=" * 50,
            "MULTIMODAL HIGHLIGHT GENERATION SUMMARY",
            "=" * 50,
            f"Video Duration: {result.metadata.get('duration', 0):.2f}s",
            f"Highlights Found: {len(result.segments)}",
            f"Total Highlight Duration: {result.metadata.get('total_highlight_duration', 0):.2f}s",
            "",
            "HIGHLIGHTS:",
            "-" * 50,
        ]

        for i, seg in enumerate(result.segments):
            lines.extend([
                f"\n[Highlight {i+1}]",
                f"  Time: {seg.start_time:.2f}s - {seg.end_time:.2f}s ({seg.duration:.2f}s)",
                f"  Score: {seg.score:.3f}",
                f"  Peak: {seg.peak_time:.2f}s",
                f"  Contributions:",
                f"    Audio: {seg.audio_contribution:.3f}",
                f"    Video: {seg.video_contribution:.3f}",
                f"    Text:  {seg.text_contribution:.3f}",
            ])
            if seg.label:
                lines.append(f"  Label: {seg.label}")

        lines.extend([
            "",
            "=" * 50,
        ])

        return "\n".join(lines)
