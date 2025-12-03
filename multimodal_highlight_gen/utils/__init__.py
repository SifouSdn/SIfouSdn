"""
Utility functions for the multimodal highlight generation system.
"""

from .video_utils import extract_audio_from_video, load_video_frames
from .segment_utils import create_highlight_segments, merge_overlapping_segments

__all__ = [
    "extract_audio_from_video",
    "load_video_frames",
    "create_highlight_segments",
    "merge_overlapping_segments",
]
