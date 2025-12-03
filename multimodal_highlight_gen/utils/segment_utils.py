"""
Segment utility functions for highlight generation.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Constants for segment adjustment
CENTER_ADJUSTMENT_FACTOR = 2  # Factor for centering excess duration reduction


@dataclass
class HighlightSegment:
    """Represents a highlight segment."""

    start_time: float  # seconds
    end_time: float  # seconds
    score: float  # 0.0 to 1.0
    peak_time: float  # timestamp of peak excitement
    audio_contribution: float
    video_contribution: float
    text_contribution: float
    label: Optional[str] = None

    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end_time - self.start_time


def create_highlight_segments(
    timestamps: np.ndarray,
    scores: np.ndarray,
    min_score: float = 0.6,
    min_duration: float = 3.0,
    max_duration: float = 30.0,
    context_before: float = 2.0,
    context_after: float = 5.0,
    audio_scores: Optional[np.ndarray] = None,
    video_scores: Optional[np.ndarray] = None,
    text_scores: Optional[np.ndarray] = None,
) -> List[HighlightSegment]:
    """
    Create highlight segments from time-series scores.

    Args:
        timestamps: Array of timestamps
        scores: Array of highlight scores (0-1)
        min_score: Minimum score to consider as highlight
        min_duration: Minimum segment duration in seconds
        max_duration: Maximum segment duration in seconds
        context_before: Seconds of context before peak
        context_after: Seconds of context after peak
        audio_scores: Optional per-timestamp audio contribution
        video_scores: Optional per-timestamp video contribution
        text_scores: Optional per-timestamp text contribution

    Returns:
        List of HighlightSegment objects
    """
    if len(timestamps) == 0 or len(scores) == 0:
        return []

    segments = []

    # Find peaks above threshold
    above_threshold = scores >= min_score

    # Find contiguous regions
    in_segment = False
    segment_start_idx = 0

    for i, is_above in enumerate(above_threshold):
        if is_above and not in_segment:
            segment_start_idx = i
            in_segment = True
        elif not is_above and in_segment:
            # End of segment
            segment = _create_segment(
                timestamps,
                scores,
                segment_start_idx,
                i,
                context_before,
                context_after,
                max_duration,
                audio_scores,
                video_scores,
                text_scores,
            )
            if segment.duration >= min_duration:
                segments.append(segment)
            in_segment = False

    # Handle segment at end
    if in_segment:
        segment = _create_segment(
            timestamps,
            scores,
            segment_start_idx,
            len(timestamps),
            context_before,
            context_after,
            max_duration,
            audio_scores,
            video_scores,
            text_scores,
        )
        if segment.duration >= min_duration:
            segments.append(segment)

    logger.info(f"Created {len(segments)} highlight segments")
    return segments


def _create_segment(
    timestamps: np.ndarray,
    scores: np.ndarray,
    start_idx: int,
    end_idx: int,
    context_before: float,
    context_after: float,
    max_duration: float,
    audio_scores: Optional[np.ndarray],
    video_scores: Optional[np.ndarray],
    text_scores: Optional[np.ndarray],
) -> HighlightSegment:
    """Create a single highlight segment."""
    # Find peak within segment
    segment_scores = scores[start_idx:end_idx]
    peak_idx_in_segment = np.argmax(segment_scores)
    peak_idx = start_idx + peak_idx_in_segment
    peak_time = timestamps[peak_idx]
    peak_score = scores[peak_idx]

    # Calculate start and end with context
    start_time = max(0, peak_time - context_before)
    end_time = peak_time + context_after

    # Enforce max duration
    if end_time - start_time > max_duration:
        excess = (end_time - start_time) - max_duration
        start_time += excess / CENTER_ADJUSTMENT_FACTOR
        end_time -= excess / CENTER_ADJUSTMENT_FACTOR

    # Calculate contributions
    audio_contribution = 0.0
    video_contribution = 0.0
    text_contribution = 0.0

    if audio_scores is not None and len(audio_scores) > peak_idx:
        audio_contribution = float(audio_scores[peak_idx])
    if video_scores is not None and len(video_scores) > peak_idx:
        video_contribution = float(video_scores[peak_idx])
    if text_scores is not None and len(text_scores) > peak_idx:
        text_contribution = float(text_scores[peak_idx])

    return HighlightSegment(
        start_time=start_time,
        end_time=end_time,
        score=float(peak_score),
        peak_time=peak_time,
        audio_contribution=audio_contribution,
        video_contribution=video_contribution,
        text_contribution=text_contribution,
    )


def merge_overlapping_segments(
    segments: List[HighlightSegment],
    merge_threshold: float = 5.0,
) -> List[HighlightSegment]:
    """
    Merge overlapping or close segments.

    Args:
        segments: List of highlight segments
        merge_threshold: Maximum gap between segments to merge (seconds)

    Returns:
        List of merged segments
    """
    if not segments:
        return []

    # Sort by start time
    sorted_segments = sorted(segments, key=lambda s: s.start_time)

    merged = [sorted_segments[0]]

    for segment in sorted_segments[1:]:
        last = merged[-1]

        # Check if segments overlap or are close enough to merge
        gap = segment.start_time - last.end_time
        if gap <= merge_threshold:
            # Merge segments
            merged[-1] = HighlightSegment(
                start_time=last.start_time,
                end_time=max(last.end_time, segment.end_time),
                score=max(last.score, segment.score),
                peak_time=last.peak_time
                if last.score >= segment.score
                else segment.peak_time,
                audio_contribution=max(
                    last.audio_contribution, segment.audio_contribution
                ),
                video_contribution=max(
                    last.video_contribution, segment.video_contribution
                ),
                text_contribution=max(
                    last.text_contribution, segment.text_contribution
                ),
                label=last.label or segment.label,
            )
        else:
            merged.append(segment)

    logger.info(
        f"Merged {len(segments)} segments into {len(merged)} segments"
    )
    return merged


def select_top_segments(
    segments: List[HighlightSegment],
    max_segments: Optional[int] = None,
    max_total_duration: Optional[float] = None,
) -> List[HighlightSegment]:
    """
    Select top segments by score.

    Args:
        segments: List of highlight segments
        max_segments: Maximum number of segments to select
        max_total_duration: Maximum total duration of selected segments

    Returns:
        List of selected segments (sorted by time)
    """
    if not segments:
        return []

    # Sort by score descending
    sorted_by_score = sorted(segments, key=lambda s: s.score, reverse=True)

    selected = []
    total_duration = 0.0

    for segment in sorted_by_score:
        # Check max segments
        if max_segments and len(selected) >= max_segments:
            break

        # Check max duration
        if max_total_duration:
            if total_duration + segment.duration > max_total_duration:
                continue

        selected.append(segment)
        total_duration += segment.duration

    # Sort selected by time
    selected.sort(key=lambda s: s.start_time)

    logger.info(
        f"Selected {len(selected)} segments "
        f"(total duration: {total_duration:.2f}s)"
    )
    return selected
