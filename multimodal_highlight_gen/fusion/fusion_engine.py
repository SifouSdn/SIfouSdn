"""
Multimodal Fusion Engine.

This module combines signals from audio, video, and text analyzers
to produce unified highlight scores using various fusion strategies.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FusionMethod(Enum):
    """Available fusion methods."""

    WEIGHTED_AVERAGE = "weighted_average"
    ATTENTION = "attention"
    LATE_FUSION = "late_fusion"
    MAX_POOLING = "max_pooling"


@dataclass
class FusionResult:
    """Result of multimodal fusion."""

    timestamps: np.ndarray
    fused_scores: np.ndarray
    audio_scores: np.ndarray
    video_scores: np.ndarray
    text_scores: np.ndarray
    confidence: np.ndarray


# Constants for fusion algorithms
ATTENTION_TEMPERATURE = 2.0  # Temperature scaling for attention-based fusion


class MultimodalFusionEngine:
    """
    Fuses multimodal signals for highlight detection.

    Combines audio, video, and text excitement signals using
    configurable fusion strategies to produce unified highlight scores.
    """

    def __init__(
        self,
        audio_weight: float = 0.35,
        video_weight: float = 0.30,
        text_weight: float = 0.35,
        fusion_method: str = "weighted_average",
        temporal_smoothing_window: float = 3.0,
        min_highlight_score: float = 0.65,
    ):
        """
        Initialize the fusion engine.

        Args:
            audio_weight: Weight for audio modality (0-1)
            video_weight: Weight for video modality (0-1)
            text_weight: Weight for text modality (0-1)
            fusion_method: Fusion strategy to use
            temporal_smoothing_window: Window size for temporal smoothing (seconds)
            min_highlight_score: Minimum score for highlight consideration
        """
        # Normalize weights
        total_weight = audio_weight + video_weight + text_weight
        self.audio_weight = audio_weight / total_weight
        self.video_weight = video_weight / total_weight
        self.text_weight = text_weight / total_weight

        self.fusion_method = FusionMethod(fusion_method)
        self.temporal_smoothing_window = temporal_smoothing_window
        self.min_highlight_score = min_highlight_score

        logger.info(
            f"Initialized fusion engine: method={fusion_method}, "
            f"weights=[audio={self.audio_weight:.2f}, "
            f"video={self.video_weight:.2f}, text={self.text_weight:.2f}]"
        )

    def fuse(
        self,
        audio_timestamps: np.ndarray,
        audio_scores: np.ndarray,
        video_timestamps: np.ndarray,
        video_scores: np.ndarray,
        text_timestamps: np.ndarray,
        text_scores: np.ndarray,
        time_resolution: float = 0.5,
    ) -> FusionResult:
        """
        Fuse multimodal signals into unified highlight scores.

        Args:
            audio_timestamps: Timestamps for audio scores
            audio_scores: Audio excitement scores (0-1)
            video_timestamps: Timestamps for video scores
            video_scores: Video excitement scores (0-1)
            text_timestamps: Timestamps for text scores
            text_scores: Text excitement scores (0-1)
            time_resolution: Output time resolution in seconds

        Returns:
            FusionResult with fused scores
        """
        # Determine common timeline
        max_time = max(
            audio_timestamps[-1] if len(audio_timestamps) > 0 else 0,
            video_timestamps[-1] if len(video_timestamps) > 0 else 0,
            text_timestamps[-1] if len(text_timestamps) > 0 else 0,
        )

        if max_time == 0:
            return FusionResult(
                timestamps=np.array([]),
                fused_scores=np.array([]),
                audio_scores=np.array([]),
                video_scores=np.array([]),
                text_scores=np.array([]),
                confidence=np.array([]),
            )

        # Create unified timeline
        unified_timestamps = np.arange(0, max_time, time_resolution)
        n_frames = len(unified_timestamps)

        # Interpolate each modality to unified timeline
        audio_interp = self._interpolate_scores(
            audio_timestamps, audio_scores, unified_timestamps
        )
        video_interp = self._interpolate_scores(
            video_timestamps, video_scores, unified_timestamps
        )
        text_interp = self._interpolate_scores(
            text_timestamps, text_scores, unified_timestamps
        )

        # Apply fusion based on method
        if self.fusion_method == FusionMethod.WEIGHTED_AVERAGE:
            fused = self._weighted_average_fusion(
                audio_interp, video_interp, text_interp
            )
        elif self.fusion_method == FusionMethod.ATTENTION:
            fused = self._attention_fusion(
                audio_interp, video_interp, text_interp
            )
        elif self.fusion_method == FusionMethod.LATE_FUSION:
            fused = self._late_fusion(
                audio_interp, video_interp, text_interp
            )
        elif self.fusion_method == FusionMethod.MAX_POOLING:
            fused = self._max_pooling_fusion(
                audio_interp, video_interp, text_interp
            )
        else:
            fused = self._weighted_average_fusion(
                audio_interp, video_interp, text_interp
            )

        # Apply temporal smoothing
        if self.temporal_smoothing_window > 0:
            fused = self._apply_temporal_smoothing(
                fused, time_resolution
            )

        # Compute confidence based on modality agreement
        confidence = self._compute_confidence(
            audio_interp, video_interp, text_interp, fused
        )

        return FusionResult(
            timestamps=unified_timestamps,
            fused_scores=fused,
            audio_scores=audio_interp,
            video_scores=video_interp,
            text_scores=text_interp,
            confidence=confidence,
        )

    def _interpolate_scores(
        self,
        timestamps: np.ndarray,
        scores: np.ndarray,
        target_timestamps: np.ndarray,
    ) -> np.ndarray:
        """Interpolate scores to target timestamps."""
        if len(timestamps) == 0 or len(scores) == 0:
            return np.zeros(len(target_timestamps))

        # Use linear interpolation with zero padding outside range
        interpolated = np.interp(
            target_timestamps,
            timestamps,
            scores,
            left=0.0,
            right=0.0,
        )

        return interpolated

    def _weighted_average_fusion(
        self,
        audio: np.ndarray,
        video: np.ndarray,
        text: np.ndarray,
    ) -> np.ndarray:
        """Simple weighted average fusion."""
        return (
            self.audio_weight * audio
            + self.video_weight * video
            + self.text_weight * text
        )

    def _attention_fusion(
        self,
        audio: np.ndarray,
        video: np.ndarray,
        text: np.ndarray,
    ) -> np.ndarray:
        """
        Attention-based fusion that weighs modalities dynamically
        based on their current values.
        """
        # Stack modalities
        stacked = np.stack([audio, video, text], axis=0)  # (3, T)

        # Compute softmax attention weights per timestep
        # Higher scoring modalities get more weight
        exp_scores = np.exp(stacked * ATTENTION_TEMPERATURE)
        attention_weights = exp_scores / (
            np.sum(exp_scores, axis=0, keepdims=True) + 1e-8
        )

        # Apply base weights
        base_weights = np.array([
            [self.audio_weight],
            [self.video_weight],
            [self.text_weight],
        ])
        combined_weights = attention_weights * base_weights
        combined_weights = combined_weights / (
            np.sum(combined_weights, axis=0, keepdims=True) + 1e-8
        )

        # Weighted sum
        fused = np.sum(stacked * combined_weights, axis=0)

        return fused

    def _late_fusion(
        self,
        audio: np.ndarray,
        video: np.ndarray,
        text: np.ndarray,
    ) -> np.ndarray:
        """
        Late fusion: each modality votes independently,
        then votes are combined.
        """
        # Threshold each modality
        threshold = 0.5
        audio_vote = (audio >= threshold).astype(float)
        video_vote = (video >= threshold).astype(float)
        text_vote = (text >= threshold).astype(float)

        # Weighted voting
        votes = (
            self.audio_weight * audio_vote
            + self.video_weight * video_vote
            + self.text_weight * text_vote
        )

        # Combine votes with original scores for smoother output
        confidence = (
            self.audio_weight * audio
            + self.video_weight * video
            + self.text_weight * text
        )

        # Final score is vote strength modulated by confidence
        fused = votes * 0.5 + confidence * 0.5

        return fused

    def _max_pooling_fusion(
        self,
        audio: np.ndarray,
        video: np.ndarray,
        text: np.ndarray,
    ) -> np.ndarray:
        """
        Max pooling fusion: take maximum across modalities.
        Useful when any single modality indicating excitement is enough.
        """
        stacked = np.stack([audio, video, text], axis=0)
        return np.max(stacked, axis=0)

    def _apply_temporal_smoothing(
        self,
        scores: np.ndarray,
        time_resolution: float,
    ) -> np.ndarray:
        """Apply temporal smoothing to reduce noise."""
        window_size = int(self.temporal_smoothing_window / time_resolution)
        if window_size <= 1:
            return scores

        # Use Gaussian-like kernel for smooth transitions
        kernel = np.exp(-0.5 * np.linspace(-2, 2, window_size) ** 2)
        kernel = kernel / kernel.sum()

        smoothed = np.convolve(scores, kernel, mode="same")

        return smoothed

    def _compute_confidence(
        self,
        audio: np.ndarray,
        video: np.ndarray,
        text: np.ndarray,
        fused: np.ndarray,
    ) -> np.ndarray:
        """
        Compute confidence based on modality agreement.
        Higher confidence when modalities agree.
        """
        stacked = np.stack([audio, video, text], axis=0)

        # Confidence is inverse of standard deviation across modalities
        std = np.std(stacked, axis=0)
        max_std = 0.5  # Maximum expected std

        # Convert std to confidence (low std = high confidence)
        confidence = 1.0 - np.clip(std / max_std, 0, 1)

        # Modulate by fused score (higher scores have higher base confidence)
        confidence = confidence * 0.5 + fused * 0.5

        return confidence

    def get_highlight_candidates(
        self,
        fusion_result: FusionResult,
    ) -> List[Tuple[float, float, float]]:
        """
        Get candidate highlight timestamps from fusion result.

        Args:
            fusion_result: Result from fuse() method

        Returns:
            List of (start_time, peak_time, score) tuples
        """
        timestamps = fusion_result.timestamps
        scores = fusion_result.fused_scores

        if len(timestamps) == 0:
            return []

        candidates = []

        # Find regions above threshold
        above_threshold = scores >= self.min_highlight_score

        in_region = False
        region_start_idx = 0

        for i, is_above in enumerate(above_threshold):
            if is_above and not in_region:
                region_start_idx = i
                in_region = True
            elif not is_above and in_region:
                # End of region
                region_scores = scores[region_start_idx:i]
                peak_idx = region_start_idx + np.argmax(region_scores)

                candidates.append((
                    timestamps[region_start_idx],
                    timestamps[peak_idx],
                    float(scores[peak_idx]),
                ))
                in_region = False

        # Handle region at end
        if in_region:
            region_scores = scores[region_start_idx:]
            peak_idx = region_start_idx + np.argmax(region_scores)

            candidates.append((
                timestamps[region_start_idx],
                timestamps[peak_idx],
                float(scores[peak_idx]),
            ))

        # Sort by score descending
        candidates.sort(key=lambda x: x[2], reverse=True)

        return candidates
