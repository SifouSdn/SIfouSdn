"""
Audio Analyzer for Crowd Roar Detection.

This module analyzes audio signals from sports videos to detect
crowd roars and excitement peaks that indicate highlight-worthy moments.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AudioEvent:
    """Represents a detected audio event (e.g., crowd roar)."""

    start_time: float  # seconds
    end_time: float  # seconds
    intensity: float  # 0.0 to 1.0
    event_type: str  # e.g., "crowd_roar", "whistle", "cheer"
    confidence: float  # detection confidence


class AudioAnalyzer:
    """
    Analyzes audio signals to detect crowd roars and excitement peaks.

    Uses spectral analysis and energy-based features to identify
    moments of high crowd excitement in sports broadcasts.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 512,
        n_mels: int = 128,
        crowd_roar_threshold: float = 0.7,
        min_roar_duration: float = 1.0,
        energy_percentile: float = 90.0,
        window_size: float = 2.0,
    ):
        """
        Initialize the audio analyzer.

        Args:
            sample_rate: Target sample rate for audio processing
            hop_length: Hop length for spectral analysis
            n_mels: Number of mel bands for mel spectrogram
            crowd_roar_threshold: Threshold for crowd roar detection (0-1)
            min_roar_duration: Minimum duration for a valid roar (seconds)
            energy_percentile: Percentile for energy-based thresholding
            window_size: Analysis window size in seconds
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.crowd_roar_threshold = crowd_roar_threshold
        self.min_roar_duration = min_roar_duration
        self.energy_percentile = energy_percentile
        self.window_size = window_size

        self._model_loaded = False
        self._spectral_model = None

    def _ensure_dependencies(self):
        """Lazy load heavy dependencies."""
        try:
            import librosa

            self._librosa = librosa
        except ImportError:
            logger.warning(
                "librosa not installed. Using basic audio analysis. "
                "Install with: pip install librosa"
            )
            self._librosa = None

    def analyze(
        self,
        audio_path: Optional[str] = None,
        audio_data: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
    ) -> List[AudioEvent]:
        """
        Analyze audio for crowd roar events.

        Args:
            audio_path: Path to audio file
            audio_data: Audio data as numpy array (if already loaded)
            sr: Sample rate of audio_data (required if audio_data provided)

        Returns:
            List of detected AudioEvent objects
        """
        self._ensure_dependencies()

        # Load audio if path provided
        if audio_path is not None and self._librosa is not None:
            audio_data, sr = self._librosa.load(audio_path, sr=self.sample_rate)
        elif audio_data is None:
            raise ValueError("Either audio_path or audio_data must be provided")

        if sr is None:
            sr = self.sample_rate

        # Get excitement scores over time
        excitement_scores = self._compute_excitement_scores(audio_data, sr)

        # Detect crowd roar events
        events = self._detect_events(excitement_scores, sr)

        logger.info(f"Detected {len(events)} audio events")
        return events

    def get_excitement_timeline(
        self,
        audio_path: Optional[str] = None,
        audio_data: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
        time_resolution: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get excitement scores over time.

        Args:
            audio_path: Path to audio file
            audio_data: Audio data as numpy array
            sr: Sample rate
            time_resolution: Time resolution in seconds

        Returns:
            Tuple of (timestamps, excitement_scores)
        """
        self._ensure_dependencies()

        if audio_path is not None and self._librosa is not None:
            audio_data, sr = self._librosa.load(audio_path, sr=self.sample_rate)
        elif audio_data is None:
            raise ValueError("Either audio_path or audio_data must be provided")

        if sr is None:
            sr = self.sample_rate

        excitement_scores = self._compute_excitement_scores(audio_data, sr)

        # Resample to desired time resolution
        samples_per_frame = int(time_resolution * sr / self.hop_length)
        num_frames = len(excitement_scores)
        num_output_frames = num_frames // samples_per_frame

        resampled_scores = np.zeros(num_output_frames)
        for i in range(num_output_frames):
            start_idx = i * samples_per_frame
            end_idx = min((i + 1) * samples_per_frame, num_frames)
            resampled_scores[i] = np.max(excitement_scores[start_idx:end_idx])

        timestamps = np.arange(num_output_frames) * time_resolution

        return timestamps, resampled_scores

    def _compute_excitement_scores(
        self, audio_data: np.ndarray, sr: int
    ) -> np.ndarray:
        """
        Compute frame-wise excitement scores from audio.

        Combines multiple features:
        - RMS energy
        - Spectral centroid (brightness)
        - Spectral rolloff
        - Zero crossing rate
        """
        if self._librosa is not None:
            return self._compute_librosa_features(audio_data, sr)
        else:
            return self._compute_basic_features(audio_data, sr)

    def _compute_librosa_features(
        self, audio_data: np.ndarray, sr: int
    ) -> np.ndarray:
        """Compute features using librosa."""
        librosa = self._librosa

        # RMS Energy - indicates loudness
        rms = librosa.feature.rms(
            y=audio_data, hop_length=self.hop_length
        )[0]

        # Spectral centroid - indicates brightness/excitement
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio_data, sr=sr, hop_length=self.hop_length
        )[0]

        # Spectral rolloff - frequency below which 85% of energy is contained
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio_data, sr=sr, hop_length=self.hop_length
        )[0]

        # Zero crossing rate - can indicate noise/crowd sounds
        zcr = librosa.feature.zero_crossing_rate(
            audio_data, hop_length=self.hop_length
        )[0]

        # Normalize features
        rms_norm = self._normalize(rms)
        centroid_norm = self._normalize(spectral_centroid)
        rolloff_norm = self._normalize(spectral_rolloff)
        zcr_norm = self._normalize(zcr)

        # Combine features with weights emphasizing energy and spectral content
        excitement_scores = (
            0.4 * rms_norm
            + 0.25 * centroid_norm
            + 0.2 * rolloff_norm
            + 0.15 * zcr_norm
        )

        # Apply temporal smoothing
        window_frames = int(self.window_size * sr / self.hop_length)
        if window_frames > 1:
            kernel = np.ones(window_frames) / window_frames
            excitement_scores = np.convolve(excitement_scores, kernel, mode="same")

        return excitement_scores

    def _compute_basic_features(
        self, audio_data: np.ndarray, sr: int
    ) -> np.ndarray:
        """Compute basic features without librosa."""
        # Frame-based RMS energy
        frame_length = int(self.window_size * sr)
        hop = self.hop_length
        num_frames = max(1, (len(audio_data) - frame_length) // hop + 1)

        rms = np.zeros(num_frames)
        for i in range(num_frames):
            start = i * hop
            end = min(start + frame_length, len(audio_data))
            frame = audio_data[start:end]
            rms[i] = np.sqrt(np.mean(frame**2))

        return self._normalize(rms)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize array to 0-1 range."""
        x_min = x.min()
        x_max = x.max()
        if x_max - x_min > 0:
            return (x - x_min) / (x_max - x_min)
        return np.zeros_like(x)

    def _detect_events(
        self, excitement_scores: np.ndarray, sr: int
    ) -> List[AudioEvent]:
        """Detect crowd roar events from excitement scores."""
        events = []

        # Use dynamic threshold based on percentile
        threshold = np.percentile(excitement_scores, self.energy_percentile)
        threshold = max(threshold, self.crowd_roar_threshold)

        # Find regions above threshold
        above_threshold = excitement_scores >= threshold
        frame_duration = self.hop_length / sr

        # Group consecutive frames
        in_event = False
        event_start = 0

        for i, is_above in enumerate(above_threshold):
            if is_above and not in_event:
                event_start = i
                in_event = True
            elif not is_above and in_event:
                event_end = i
                duration = (event_end - event_start) * frame_duration

                if duration >= self.min_roar_duration:
                    start_time = event_start * frame_duration
                    end_time = event_end * frame_duration
                    intensity = np.mean(
                        excitement_scores[event_start:event_end]
                    )
                    confidence = min(1.0, intensity / threshold)

                    events.append(
                        AudioEvent(
                            start_time=start_time,
                            end_time=end_time,
                            intensity=float(intensity),
                            event_type="crowd_roar",
                            confidence=float(confidence),
                        )
                    )

                in_event = False

        # Handle event at end of audio
        if in_event:
            event_end = len(above_threshold)
            duration = (event_end - event_start) * frame_duration

            if duration >= self.min_roar_duration:
                start_time = event_start * frame_duration
                end_time = event_end * frame_duration
                intensity = np.mean(excitement_scores[event_start:event_end])
                confidence = min(1.0, intensity / threshold)

                events.append(
                    AudioEvent(
                        start_time=start_time,
                        end_time=end_time,
                        intensity=float(intensity),
                        event_type="crowd_roar",
                        confidence=float(confidence),
                    )
                )

        return events
