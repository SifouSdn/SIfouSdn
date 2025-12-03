"""
Video utility functions for multimodal highlight generation.
"""

import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def extract_audio_from_video(
    video_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = 22050,
) -> Tuple[np.ndarray, int]:
    """
    Extract audio from a video file.

    Args:
        video_path: Path to the video file
        output_path: Optional path to save extracted audio
        sample_rate: Target sample rate

    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        import librosa

        # Load audio directly from video using librosa
        audio_data, sr = librosa.load(video_path, sr=sample_rate)
        logger.info(
            f"Extracted audio: {len(audio_data)/sr:.2f}s at {sr}Hz"
        )

        if output_path:
            import soundfile as sf

            sf.write(output_path, audio_data, sr)
            logger.info(f"Saved audio to: {output_path}")

        return audio_data, sr

    except ImportError:
        logger.warning("librosa not available, trying moviepy...")

        try:
            from moviepy.editor import VideoFileClip

            video = VideoFileClip(video_path)
            audio = video.audio

            if audio is None:
                raise ValueError("Video has no audio track")

            # Get audio as numpy array
            fps = sample_rate
            audio_data = audio.to_soundarray(fps=fps)

            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            video.close()

            if output_path:
                import soundfile as sf

                sf.write(output_path, audio_data, sample_rate)

            return audio_data, sample_rate

        except ImportError:
            logger.error(
                "Neither librosa nor moviepy available. "
                "Install with: pip install librosa or pip install moviepy"
            )
            raise


def load_video_frames(
    video_path: str,
    target_fps: int = 5,
    resize: Optional[Tuple[int, int]] = None,
    max_frames: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[float], dict]:
    """
    Load video frames at a target FPS.

    Args:
        video_path: Path to the video file
        target_fps: Target frames per second to extract
        resize: Optional (width, height) to resize frames
        max_frames: Maximum number of frames to extract

    Returns:
        Tuple of (frames, timestamps, video_info)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV not installed. Install with: pip install opencv-python"
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    video_info = {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration": duration,
    }

    frame_skip = max(1, int(fps / target_fps))

    frames = []
    timestamps = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            if resize:
                frame = cv2.resize(frame, resize)

            frames.append(frame)
            timestamps.append(frame_idx / fps)

            if max_frames and len(frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()

    logger.info(
        f"Loaded {len(frames)} frames from {video_path} "
        f"(original: {total_frames} frames, {duration:.2f}s)"
    )

    return frames, timestamps, video_info


def get_video_duration(video_path: str) -> float:
    """Get the duration of a video in seconds."""
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        return total_frames / fps if fps > 0 else 0

    except ImportError:
        try:
            from moviepy.editor import VideoFileClip

            video = VideoFileClip(video_path)
            duration = video.duration
            video.close()
            return duration
        except ImportError:
            raise ImportError(
                "Neither OpenCV nor moviepy available for video processing"
            )
