"""
Multimodal Highlight Generation System

This package implements an AI-powered highlight generation system for sports videos
using multimodal sentiment analysis combining:
- Audio analysis (crowd roar detection)
- Video analysis (facial expression recognition)
- Text analysis (live commentary sentiment)

The system produces AI-generated highlights instead of using rule-based timestamps.
"""

from .pipeline import MultimodalHighlightPipeline
from .config import HighlightConfig

__version__ = "1.0.0"
__all__ = ["MultimodalHighlightPipeline", "HighlightConfig"]
