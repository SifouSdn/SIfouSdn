"""
Example usage of the Multimodal Highlight Generation system.

This script demonstrates how to use the highlight generation pipeline
with various input configurations.
"""

import os
import sys
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimodal_highlight_gen import MultimodalHighlightPipeline, HighlightConfig
from multimodal_highlight_gen.analyzers.text_analyzer import CommentaryEntry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def example_with_video_file():
    """Example: Generate highlights from a video file."""
    print("\n" + "="*60)
    print("Example 1: Generate highlights from video file")
    print("="*60)
    
    # Initialize pipeline with default configuration
    pipeline = MultimodalHighlightPipeline()
    
    # Example commentary (would normally come from subtitles or live feed)
    commentary = [
        CommentaryEntry(timestamp=10.0, text="The game is underway!"),
        CommentaryEntry(timestamp=45.0, text="GOAL! What an incredible strike!"),
        CommentaryEntry(timestamp=46.0, text="Unbelievable finish from the edge of the box!"),
        CommentaryEntry(timestamp=90.0, text="Great save by the goalkeeper!"),
        CommentaryEntry(timestamp=120.0, text="And there's the final whistle."),
    ]
    
    # Note: In production, you would use an actual video file
    # result = pipeline.generate_highlights(
    #     video_path="match.mp4",
    #     commentary=commentary
    # )
    
    print("Pipeline initialized successfully!")
    print(f"Audio analyzer: {pipeline.audio_analyzer}")
    print(f"Video analyzer: {pipeline.video_analyzer}")
    print(f"Text analyzer: {pipeline.text_analyzer}")


def example_with_custom_config():
    """Example: Generate highlights with custom configuration."""
    print("\n" + "="*60)
    print("Example 2: Custom configuration")
    print("="*60)
    
    # Create custom configuration
    config = HighlightConfig()
    
    # Adjust weights to prioritize audio (crowd reactions)
    config.fusion.audio_weight = 0.5
    config.fusion.video_weight = 0.2
    config.fusion.text_weight = 0.3
    
    # Adjust highlight parameters
    config.min_highlight_duration = 5.0
    config.max_highlight_duration = 20.0
    config.context_before = 3.0
    config.context_after = 7.0
    
    # Lower threshold for more highlights
    config.fusion.min_highlight_score = 0.5
    
    # Initialize pipeline with custom config
    pipeline = MultimodalHighlightPipeline(config)
    
    print("Custom configuration applied:")
    print(f"  Audio weight: {config.fusion.audio_weight}")
    print(f"  Video weight: {config.fusion.video_weight}")
    print(f"  Text weight: {config.fusion.text_weight}")
    print(f"  Min highlight duration: {config.min_highlight_duration}s")
    print(f"  Max highlight duration: {config.max_highlight_duration}s")


def example_text_analysis():
    """Example: Analyze commentary text."""
    print("\n" + "="*60)
    print("Example 3: Text analysis")
    print("="*60)
    
    from multimodal_highlight_gen.analyzers import TextAnalyzer
    
    analyzer = TextAnalyzer()
    
    # Example commentary lines
    texts = [
        "Normal play continues in the midfield.",
        "GOAL!!! What an incredible strike from outside the box!",
        "Great save by the goalkeeper! That was close!",
        "And the referee shows a red card!",
        "The players are walking off for halftime.",
    ]
    
    print("\nAnalyzing commentary texts:")
    print("-" * 50)
    
    for text in texts:
        result = analyzer.analyze_text(text)
        print(f"\nText: \"{text}\"")
        print(f"  Excitement score: {result['excitement_score']:.3f}")
        print(f"  Sentiment score: {result['sentiment_score']:.3f}")
        print(f"  Keywords found: {result['keywords_found']}")
        print(f"  Is explosive: {result['is_explosive']}")


def example_fusion_demo():
    """Example: Demonstrate multimodal fusion."""
    print("\n" + "="*60)
    print("Example 4: Multimodal fusion demonstration")
    print("="*60)
    
    import numpy as np
    from multimodal_highlight_gen.fusion import MultimodalFusionEngine
    
    # Create fusion engine
    fusion = MultimodalFusionEngine(
        audio_weight=0.35,
        video_weight=0.30,
        text_weight=0.35,
        fusion_method="weighted_average"
    )
    
    # Simulate modality scores over time
    timestamps = np.arange(0, 60, 0.5)  # 60 seconds, 0.5s resolution
    
    # Simulate audio: high around 20s and 45s (crowd roars)
    audio_scores = np.zeros_like(timestamps)
    audio_scores[35:45] = 0.8  # Crowd roar at ~17-22s
    audio_scores[85:95] = 0.9  # Bigger crowd roar at ~42-47s
    
    # Simulate video: high emotion around 45s
    video_scores = np.zeros_like(timestamps)
    video_scores[88:98] = 0.7  # Emotional reactions at ~44-49s
    
    # Simulate text: commentary spike at 45s
    text_scores = np.zeros_like(timestamps)
    text_scores[90:100] = 0.95  # "GOAL!" commentary at ~45-50s
    
    # Fuse modalities
    result = fusion.fuse(
        audio_timestamps=timestamps,
        audio_scores=audio_scores,
        video_timestamps=timestamps,
        video_scores=video_scores,
        text_timestamps=timestamps,
        text_scores=text_scores,
    )
    
    print(f"\nFusion complete!")
    print(f"  Timeline length: {len(result.timestamps)} frames")
    print(f"  Max fused score: {result.fused_scores.max():.3f}")
    print(f"  Mean fused score: {result.fused_scores.mean():.3f}")
    
    # Get highlight candidates
    candidates = fusion.get_highlight_candidates(result)
    print(f"\nHighlight candidates found: {len(candidates)}")
    for i, (start, peak, score) in enumerate(candidates):
        print(f"  {i+1}. Peak at {peak:.1f}s (score: {score:.3f})")


def main():
    """Run all examples."""
    print("\n" + "#"*60)
    print("# MULTIMODAL HIGHLIGHT GENERATION - EXAMPLES")
    print("#"*60)
    
    example_with_video_file()
    example_with_custom_config()
    example_text_analysis()
    example_fusion_demo()
    
    print("\n" + "#"*60)
    print("# All examples completed successfully!")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
