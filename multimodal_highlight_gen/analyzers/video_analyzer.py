"""
Video Analyzer for Facial Expression Detection.

This module analyzes video frames to detect intense facial expressions
that indicate excitement or emotional moments worth highlighting.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FacialExpressionEvent:
    """Represents a detected facial expression event."""

    start_time: float  # seconds
    end_time: float  # seconds
    dominant_emotion: str
    emotion_scores: Dict[str, float]
    intensity: float  # 0.0 to 1.0
    num_faces: int
    confidence: float


@dataclass
class FrameAnalysis:
    """Analysis result for a single frame."""

    timestamp: float
    faces_detected: int
    emotions: List[Dict[str, float]]
    intensity_score: float


class VideoAnalyzer:
    """
    Analyzes video frames to detect intense facial expressions.

    Uses face detection and emotion recognition to identify
    moments of high emotional intensity in sports broadcasts.
    """

    EMOTION_LABELS = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "sad",
        "surprise",
        "neutral",
    ]

    # Constants for emotion estimation
    BASE_EMOTION_STRENGTH = 0.5
    EMOTION_VARIANCE = 0.3

    INTENSE_EMOTIONS = ["happy", "surprise", "angry", "fear"]

    def __init__(
        self,
        target_fps: int = 5,
        face_detection_confidence: float = 0.5,
        emotion_threshold: float = 0.6,
        intense_emotions: Optional[List[str]] = None,
        batch_size: int = 16,
        resize_width: int = 640,
        resize_height: int = 480,
    ):
        """
        Initialize the video analyzer.

        Args:
            target_fps: Frames per second to analyze
            face_detection_confidence: Minimum confidence for face detection
            emotion_threshold: Threshold for emotion detection
            intense_emotions: List of emotions considered "intense"
            batch_size: Batch size for processing frames
            resize_width: Width to resize frames for processing
            resize_height: Height to resize frames for processing
        """
        self.target_fps = target_fps
        self.face_detection_confidence = face_detection_confidence
        self.emotion_threshold = emotion_threshold
        self.intense_emotions = intense_emotions or self.INTENSE_EMOTIONS
        self.batch_size = batch_size
        self.resize_width = resize_width
        self.resize_height = resize_height

        self._face_detector = None
        self._emotion_model = None
        self._cv2 = None

    def _ensure_dependencies(self):
        """Lazy load heavy dependencies."""
        try:
            import cv2

            self._cv2 = cv2
        except ImportError:
            logger.warning(
                "OpenCV not installed. Video analysis will be limited. "
                "Install with: pip install opencv-python"
            )
            self._cv2 = None

        # Try to load face cascade classifier
        if self._cv2 is not None and self._face_detector is None:
            try:
                import os
                cascade_path = os.path.join(
                    self._cv2.data.haarcascades,
                    "haarcascade_frontalface_default.xml"
                )
                self._face_detector = self._cv2.CascadeClassifier(cascade_path)
            except Exception as e:
                logger.warning(f"Could not load face detector: {e}")

    def analyze(
        self,
        video_path: Optional[str] = None,
        frames: Optional[List[np.ndarray]] = None,
        timestamps: Optional[List[float]] = None,
    ) -> List[FacialExpressionEvent]:
        """
        Analyze video for facial expression events.

        Args:
            video_path: Path to video file
            frames: Pre-extracted video frames
            timestamps: Timestamps for pre-extracted frames

        Returns:
            List of detected FacialExpressionEvent objects
        """
        self._ensure_dependencies()

        if video_path is not None and self._cv2 is not None:
            frames, timestamps = self._load_video_frames(video_path)
        elif frames is None:
            raise ValueError("Either video_path or frames must be provided")

        if timestamps is None:
            # Assume frames are at target_fps
            timestamps = [i / self.target_fps for i in range(len(frames))]

        # Analyze each frame
        frame_analyses = self._analyze_frames(frames, timestamps)

        # Detect expression events from frame analyses
        events = self._detect_events(frame_analyses)

        logger.info(f"Detected {len(events)} facial expression events")
        return events

    def get_intensity_timeline(
        self,
        video_path: Optional[str] = None,
        frames: Optional[List[np.ndarray]] = None,
        timestamps: Optional[List[float]] = None,
        time_resolution: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get emotional intensity scores over time.

        Args:
            video_path: Path to video file
            frames: Pre-extracted video frames
            timestamps: Timestamps for frames
            time_resolution: Time resolution in seconds

        Returns:
            Tuple of (timestamps, intensity_scores)
        """
        self._ensure_dependencies()

        if video_path is not None and self._cv2 is not None:
            frames, timestamps = self._load_video_frames(video_path)
        elif frames is None:
            raise ValueError("Either video_path or frames must be provided")

        if timestamps is None:
            timestamps = [i / self.target_fps for i in range(len(frames))]

        frame_analyses = self._analyze_frames(frames, timestamps)

        # Extract intensity scores
        times = np.array([fa.timestamp for fa in frame_analyses])
        intensities = np.array([fa.intensity_score for fa in frame_analyses])

        # Resample to desired time resolution
        if len(times) == 0:
            return np.array([]), np.array([])

        max_time = times[-1]
        output_times = np.arange(0, max_time, time_resolution)
        output_intensities = np.zeros_like(output_times)

        for i, t in enumerate(output_times):
            # Find frames within time window
            mask = (times >= t) & (times < t + time_resolution)
            if mask.any():
                output_intensities[i] = np.max(intensities[mask])

        return output_times, output_intensities

    def _load_video_frames(
        self, video_path: str
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Load video frames at target FPS."""
        cv2 = self._cv2
        if cv2 is None:
            raise RuntimeError("OpenCV not available")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = max(1, int(fps / self.target_fps))

        frames = []
        timestamps = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                # Resize frame
                frame = cv2.resize(
                    frame, (self.resize_width, self.resize_height)
                )
                frames.append(frame)
                timestamps.append(frame_idx / fps)

            frame_idx += 1

        cap.release()
        return frames, timestamps

    def _analyze_frames(
        self, frames: List[np.ndarray], timestamps: List[float]
    ) -> List[FrameAnalysis]:
        """Analyze multiple frames for facial expressions."""
        analyses = []

        for frame, timestamp in zip(frames, timestamps):
            analysis = self._analyze_single_frame(frame, timestamp)
            analyses.append(analysis)

        return analyses

    def _analyze_single_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> FrameAnalysis:
        """Analyze a single frame for facial expressions."""
        if self._cv2 is None or self._face_detector is None:
            # Return default analysis if no CV2
            return FrameAnalysis(
                timestamp=timestamp,
                faces_detected=0,
                emotions=[],
                intensity_score=0.0,
            )

        cv2 = self._cv2

        # Convert to grayscale for face detection
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Detect faces
        faces = self._face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        num_faces = len(faces)
        emotions_list = []
        intensity_score = 0.0

        if num_faces > 0:
            # For each detected face, simulate emotion analysis
            # In a full implementation, this would use a deep learning model
            for x, y, w, h in faces:
                face_roi = gray[y : y + h, x : x + w]

                # Compute basic features as proxy for emotions
                emotions = self._estimate_emotions(face_roi)
                emotions_list.append(emotions)

            # Compute overall intensity from emotions
            intensity_score = self._compute_intensity(emotions_list)

        return FrameAnalysis(
            timestamp=timestamp,
            faces_detected=num_faces,
            emotions=emotions_list,
            intensity_score=intensity_score,
        )

    def _estimate_emotions(self, face_roi: np.ndarray) -> Dict[str, float]:
        """
        Estimate emotions from face region.

        Note: In a production system, this would use a trained emotion
        recognition model (e.g., FER2013 trained CNN). Here we use
        simple image statistics as a placeholder.
        """
        if face_roi.size == 0:
            return {emotion: 0.0 for emotion in self.EMOTION_LABELS}

        # Compute basic image statistics
        mean_intensity = np.mean(face_roi) / 255.0
        std_intensity = np.std(face_roi) / 255.0

        # Use statistics to estimate emotions (simplified)
        # Higher variance often correlates with expressive faces
        expressiveness = min(1.0, std_intensity * 3)

        emotions = {}
        for i, emotion in enumerate(self.EMOTION_LABELS):
            if emotion == "neutral":
                emotions[emotion] = max(0, 1.0 - expressiveness)
            elif emotion in self.intense_emotions:
                # Intense emotions more likely with high expressiveness
                emotions[emotion] = expressiveness * (
                    self.BASE_EMOTION_STRENGTH
                    + np.random.random() * self.EMOTION_VARIANCE
                )
            else:
                emotions[emotion] = (
                    expressiveness * np.random.random() * self.EMOTION_VARIANCE
                )

        # Normalize to sum to 1
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v / total for k, v in emotions.items()}

        return emotions

    def _compute_intensity(
        self, emotions_list: List[Dict[str, float]]
    ) -> float:
        """Compute overall emotional intensity from emotion scores."""
        if not emotions_list:
            return 0.0

        intensities = []
        for emotions in emotions_list:
            # Sum intense emotion probabilities
            intense_sum = sum(
                emotions.get(e, 0.0) for e in self.intense_emotions
            )
            intensities.append(intense_sum)

        return np.mean(intensities)

    def _detect_events(
        self, frame_analyses: List[FrameAnalysis]
    ) -> List[FacialExpressionEvent]:
        """Detect facial expression events from frame analyses."""
        events = []

        if not frame_analyses:
            return events

        # Find regions of high intensity
        in_event = False
        event_start_idx = 0

        for i, analysis in enumerate(frame_analyses):
            is_intense = analysis.intensity_score >= self.emotion_threshold

            if is_intense and not in_event:
                event_start_idx = i
                in_event = True
            elif not is_intense and in_event:
                # End of event
                event = self._create_event(
                    frame_analyses[event_start_idx:i]
                )
                if event is not None:
                    events.append(event)
                in_event = False

        # Handle event at end
        if in_event:
            event = self._create_event(
                frame_analyses[event_start_idx:]
            )
            if event is not None:
                events.append(event)

        return events

    def _create_event(
        self, analyses: List[FrameAnalysis]
    ) -> Optional[FacialExpressionEvent]:
        """Create a FacialExpressionEvent from frame analyses."""
        if not analyses:
            return None

        start_time = analyses[0].timestamp
        end_time = analyses[-1].timestamp

        # Aggregate emotions
        all_emotions: Dict[str, List[float]] = {
            e: [] for e in self.EMOTION_LABELS
        }

        total_faces = 0
        intensities = []

        for analysis in analyses:
            for emotions in analysis.emotions:
                for emotion, score in emotions.items():
                    if emotion in all_emotions:
                        all_emotions[emotion].append(score)
            total_faces += analysis.faces_detected
            intensities.append(analysis.intensity_score)

        # Compute average emotion scores
        avg_emotions = {}
        for emotion, scores in all_emotions.items():
            avg_emotions[emotion] = np.mean(scores) if scores else 0.0

        # Find dominant emotion
        dominant_emotion = max(avg_emotions.keys(), key=lambda k: avg_emotions[k])

        avg_intensity = np.mean(intensities)
        avg_faces = total_faces / len(analyses)

        return FacialExpressionEvent(
            start_time=start_time,
            end_time=end_time,
            dominant_emotion=dominant_emotion,
            emotion_scores=avg_emotions,
            intensity=float(avg_intensity),
            num_faces=int(avg_faces),
            confidence=float(min(1.0, avg_intensity)),
        )
