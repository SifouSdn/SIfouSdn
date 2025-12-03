# 👋 Hi, I’m Seif-Allah Saidoun

🎓 I’m a final year engineering student at the **National Higher School of Artificial Intelligence (ENSIA), Algiers**, pursuing a state-certified degree in **Artificial Intelligence and Data Science**.

💡 My interests lie at the intersection of **AI**, **Natural Language Processing**, **Computer Vision**, and **Cybersecurity**. I’m passionate about building intelligent systems that solve real-world problems and make technology more accessible, secure, and impactful.

🚀 I’ve worked on:
- An NLP-powered educational platform using **LLMs** and **Retrieval-Augmented Generation** for teaching Algerian history
- Full-stack web & mobile applications using **React Native**, **Spring Boot**, and **RESTful APIs**
- Machine learning models for **time series forecasting** and **document classification**
- Academic projects in **wireless systems**, **reinforcement learning**, and **cybersecurity frameworks**

🛠️ Tech I use:
`Python` | `Java` | `JavaScript` | `SQL` | `TensorFlow` | `PyTorch` | `scikit-learn` | `OpenCV` | `React Native` | `Docker` | `Git` | `Linux`

📫 Feel free to reach out for collaborations, internships, or hackathon teams:
- ✉️ Email: seif-allah.saidoun@ensia.edu.dz
- 🔗 [LinkedIn](https://www.linkedin.com/in/seif-allah-saidoun-116646246/)
- 🗂️ [My CV (Google Drive)](https://drive.google.com/file/d/1tGIgZcI6YAAYHIK7jrbFJfdOqB1X-iZA/view?usp=drive_link)

---

Thanks for visiting my GitHub profile! Let’s connect and build something meaningful 🤝


# 🎥⚽ Multimodal Highlight Generation System

An AI-powered system for automatic highlight extraction from sports videos using **Multimodal Sentiment Analysis (MSA)**. This system combines audio, video, and text signals to detect exciting moments and generate highlights instead of using rule-based timestamps.

## 🌟 Features

- **🔊 Audio Analysis (Crowd Roar Detection)**
  - Spectral analysis using mel-spectrograms
  - Energy-based crowd excitement detection
  - Temporal pattern recognition for sustained reactions

- **👁️ Video Analysis (Facial Expression Detection)**
  - Face detection and tracking
  - Emotion recognition (happy, surprise, anger, fear, etc.)
  - Intensity scoring for emotional reactions

- **📝 Text Analysis (Live Commentary Sentiment)**
  - Transformer-based sentiment analysis
  - Sports-specific keyword detection
  - Excitement scoring for explosive reactions

- **🔀 Multimodal Fusion**
  - Multiple fusion strategies (weighted average, attention, late fusion)
  - Temporal smoothing for stable predictions
  - Confidence scoring based on modality agreement

## 📁 Project Structure

```
multimodal_highlight_gen/
├── __init__.py              # Package initialization
├── config.py                # Configuration dataclasses
├── pipeline.py              # Main orchestration pipeline
├── analyzers/
│   ├── __init__.py
│   ├── audio_analyzer.py    # Crowd roar detection
│   ├── video_analyzer.py    # Facial expression analysis
│   └── text_analyzer.py     # Commentary sentiment analysis
├── fusion/
│   ├── __init__.py
│   └── fusion_engine.py     # Multimodal fusion strategies
└── utils/
    ├── __init__.py
    ├── video_utils.py       # Video/audio extraction utilities
    └── segment_utils.py     # Highlight segment management

examples/
└── demo.py                  # Usage examples and demonstrations
```

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/SifouSdn/SIfouSdn.git
cd SIfouSdn

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- **numpy** - Numerical computing
- **librosa** - Audio analysis
- **opencv-python** - Video processing
- **transformers** - NLP models for sentiment analysis
- **torch** - Deep learning framework
- **moviepy** - Video editing and export

## 📖 Usage

### Basic Usage

```python
from multimodal_highlight_gen import MultimodalHighlightPipeline, HighlightConfig
from multimodal_highlight_gen.analyzers.text_analyzer import CommentaryEntry

# Initialize pipeline
pipeline = MultimodalHighlightPipeline()

# Prepare commentary data
commentary = [
    CommentaryEntry(timestamp=45.0, text="GOAL! What an incredible strike!"),
    CommentaryEntry(timestamp=90.0, text="Great save by the goalkeeper!"),
]

# Generate highlights
result = pipeline.generate_highlights(
    video_path="match.mp4",
    commentary=commentary
)

# Print summary
print(pipeline.get_summary(result))

# Export highlights as separate video files
output_paths = pipeline.export_highlights(
    result=result,
    video_path="match.mp4",
    output_dir="./highlights"
)
```

### Custom Configuration

```python
from multimodal_highlight_gen import HighlightConfig

# Create custom configuration
config = HighlightConfig()

# Adjust modality weights
config.fusion.audio_weight = 0.5   # Prioritize crowd reactions
config.fusion.video_weight = 0.2
config.fusion.text_weight = 0.3

# Adjust highlight parameters
config.min_highlight_duration = 5.0   # Minimum 5 second highlights
config.max_highlight_duration = 20.0  # Maximum 20 second highlights
config.fusion.min_highlight_score = 0.5  # Lower threshold for more highlights

# Use custom config
pipeline = MultimodalHighlightPipeline(config)
```

### Individual Modality Analysis

```python
from multimodal_highlight_gen.analyzers import AudioAnalyzer, TextAnalyzer

# Analyze audio only
audio_analyzer = AudioAnalyzer()
timestamps, scores = audio_analyzer.get_excitement_timeline(audio_path="match.mp3")

# Analyze text only
text_analyzer = TextAnalyzer()
result = text_analyzer.analyze_text("GOAL!!! What an incredible strike!")
print(f"Excitement: {result['excitement_score']:.3f}")
print(f"Keywords: {result['keywords_found']}")
```

### Fusion Strategies

```python
from multimodal_highlight_gen.fusion import MultimodalFusionEngine

# Weighted average (default)
fusion = MultimodalFusionEngine(fusion_method="weighted_average")

# Attention-based fusion (dynamic weighting)
fusion = MultimodalFusionEngine(fusion_method="attention")

# Late fusion (voting-based)
fusion = MultimodalFusionEngine(fusion_method="late_fusion")

# Max pooling (any modality triggers)
fusion = MultimodalFusionEngine(fusion_method="max_pooling")
```

## 🎯 How It Works

### 1. Audio Analysis
The audio analyzer processes the soundtrack to detect crowd reactions:
- Extracts spectral features (RMS energy, spectral centroid, rolloff)
- Identifies sustained high-energy regions indicating crowd excitement
- Outputs time-series excitement scores

### 2. Video Analysis
The video analyzer detects emotional expressions in faces:
- Uses Haar Cascade for face detection
- Estimates emotional intensity from facial features
- Aggregates scores from multiple detected faces

### 3. Text Analysis
The text analyzer processes live commentary:
- Uses transformer-based sentiment analysis (DistilBERT)
- Detects sports-specific excitement keywords
- Scores text for explosive reactions

### 4. Multimodal Fusion
The fusion engine combines all modalities:
- Interpolates scores to a common timeline
- Applies configurable fusion strategy
- Performs temporal smoothing for stability
- Outputs unified highlight scores

### 5. Highlight Generation
The pipeline creates final highlights:
- Identifies regions above threshold
- Adds context before/after peaks
- Merges overlapping segments
- Selects top highlights based on score

## ⚙️ Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `audio.sample_rate` | Audio sample rate | 22050 |
| `audio.crowd_roar_threshold` | Threshold for crowd detection | 0.7 |
| `video.target_fps` | Frames per second to analyze | 5 |
| `video.emotion_threshold` | Threshold for emotion detection | 0.6 |
| `text.explosion_threshold` | Threshold for explosive text | 0.8 |
| `fusion.audio_weight` | Weight for audio modality | 0.35 |
| `fusion.video_weight` | Weight for video modality | 0.30 |
| `fusion.text_weight` | Weight for text modality | 0.35 |
| `fusion.fusion_method` | Fusion strategy | "weighted_average" |
| `min_highlight_duration` | Minimum highlight length | 3.0s |
| `max_highlight_duration` | Maximum highlight length | 30.0s |
| `context_before` | Context before peak | 2.0s |
| `context_after` | Context after peak | 5.0s |

## 🎯 Impact & Applications

- **🚀 Media Tech Companies** - Automate highlight generation for sports broadcasting
- **📱 Sports Apps** - Real-time highlight detection for fan engagement
- **🎬 Content Creation** - AI-assisted video editing for sports content
- **📊 Analytics** - Quantify exciting moments in match analysis

## 🔬 Technical Challenges

This project addresses key **Multimodal Sentiment Analysis (MSA)** challenges:
- **Temporal Alignment** - Synchronizing signals from different modalities
- **Modality Weighting** - Balancing contributions from audio, video, and text
- **Fusion Strategies** - Combining heterogeneous signals effectively
- **Real-time Processing** - Efficient analysis for live applications

