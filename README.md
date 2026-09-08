# NapSense

**Real-Time Driver Drowsiness Detection using Computer Vision**

> **One yawn could be the warning you need.**

NapSense is a real-time computer vision system designed to detect signs of driver drowsiness using a standard webcam. Instead of relying on computationally expensive deep-learning models, NapSense combines **facial landmarks, eye/mouth geometry, facial texture analysis, and temporal behavior** to identify potential fatigue-related behavior in real time.

The system continuously monitors facial behavior and triggers an alert when drowsiness-related patterns persist across multiple frames.

---

## Table of Contents

- [Motivation](#motivation)
- [Features](#features)
- [System Overview](#system-overview)
- [How It Works](#how-it-works)
  - [Eye Aspect Ratio (EAR)](#eye-aspect-ratio-ear)
  - [Mouth Aspect Ratio (MAR)](#mouth-aspect-ratio-mar)
  - [Local Binary Patterns (LBP)](#local-binary-patterns-lbp)
  - [Temporal Behavior Analysis](#temporal-behavior-analysis)
  - [Why Multiple Visual Signals?](#why-multiple-visual-signals)
- [Technology Stack](#technology-stack)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Evaluation Metrics](#evaluation-metrics)
- [Future Improvements](#future-improvements)
- [Key Takeaways](#key-takeaways)
- [Limitations](#limitations)
- [License](#license)

---

## Motivation

Driver drowsiness is an important road-safety problem. A moment of prolonged eye closure or a series of yawns can be an early indication of fatigue.

This project explores a practical question:

> **Can lightweight computer vision techniques provide real-time drowsiness detection using nothing more than a webcam?**

Rather than building a large neural network, NapSense combines multiple interpretable computer-vision signals to create a lightweight and deployable solution.

---

## Features

- Real-time webcam monitoring
- Eye closure detection using **Eye Aspect Ratio (EAR)**
- Yawning detection using **Mouth Aspect Ratio (MAR)**
- Facial landmark tracking using **MediaPipe Face Mesh**
- Facial texture analysis using **Local Binary Patterns (LBP)**
- Temporal analysis across consecutive frames
- Real-time drowsiness alerts
- Lightweight and CPU-friendly design
- Interpretable visual features instead of a black-box prediction alone

---

## System Overview

```
                         Webcam
                            │
                            ▼
                    Frame Acquisition
                            │
                            ▼
                   Face Landmark Detection
                            │
                    MediaPipe Face Mesh
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
         Eye Region     Mouth Region   Face Texture
              │             │             │
              ▼             ▼             ▼
             EAR           MAR            LBP
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                    Temporal Analysis
                            │
                            ▼
                  Drowsiness Assessment
                            │
                     ┌──────┴──────┐
                     ▼             ▼
                   Normal        Drowsy
                                   │
                                   ▼
                                 Alert
```

---

## How It Works

### Eye Aspect Ratio (EAR)

Eye closure is one of the primary visual indicators used by NapSense.

Facial landmarks around each eye are used to calculate the Eye Aspect Ratio (EAR).

**Formula:**

```
              ||p2 - p6|| + ||p3 - p5||
EAR =         ─────────────────────────
                    2 × ||p1 - p4||
```

Where the points represent landmarks around the eye.

- **Open eye**: The vertical distance between the eyelids remains relatively large
- **Closed eye**: The vertical distance between the eyelids decreases → EAR ↓

**Key Insight:** If the EAR remains below a configured threshold for a sustained period, the system considers the behavior a potential drowsiness event. This is more useful than treating a single low-EAR frame as drowsiness because normal blinking also causes temporary reductions in EAR.

---

### Mouth Aspect Ratio (MAR)

Yawning is another behavioral indicator of fatigue.

NapSense uses facial landmarks around the mouth to calculate the Mouth Aspect Ratio (MAR).

**Formula:**

```
              Vertical Mouth Distance
MAR =         ────────────────────────
              Horizontal Mouth Width
```

When the mouth opens significantly for a sustained period, the system can identify the behavior as a potential yawn. Again, the system considers multiple consecutive frames instead of relying on a single observation.

---

### Local Binary Patterns (LBP)

NapSense also incorporates Local Binary Patterns (LBP) for facial texture analysis.

LBP describes local image texture by comparing neighboring pixels with a central pixel. This creates a compact representation of local texture patterns.

**Process Flow:**

```
Facial Region
     │
     ▼
Grayscale / Texture Processing
     │
     ▼
Local Neighborhood Comparison
     │
     ▼
LBP Features
```

The purpose of incorporating LBP is to provide an additional visual signal alongside geometric measurements such as EAR and MAR.

---

### Temporal Behavior Analysis

One of the key design decisions in NapSense is treating drowsiness as a temporal behavioral pattern rather than a single-frame classification problem.

**Example Timeline:**

```
Frame 1 → Eyes Open
Frame 2 → Eyes Open
Frame 3 → Eyes Closing
Frame 4 → Eyes Closed
Frame 5 → Eyes Closed
Frame 6 → Eyes Closed
Frame 7 → Drowsiness Alert
```

This helps distinguish:

```
Normal Blink        vs.        Prolonged Eye Closure
```

Similarly, yawning is evaluated across consecutive frames rather than triggering an alert immediately when MAR exceeds a threshold.

---

### Why Multiple Visual Signals?

A single feature can produce false positives. For example:

- A person may blink slowly
- A person may temporarily look down
- A person may yawn without being significantly fatigued
- Lighting can affect facial appearance
- Head movement can temporarily alter landmark measurements

NapSense therefore combines:

```
    EAR
     +
    MAR
     +
    LBP
     +
Temporal Behavior
```

to obtain a richer representation of potential drowsiness.

---

## Detection Pipeline

```
┌───────────────────────┐
│      Video Frame      │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Face / Landmark       │
│ Detection             │
└───────────┬───────────┘
            │
            ▼
┌─────────────────────────────────┐
│ Feature Extraction              │
│                                 │
│  EAR     MAR     LBP            │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Temporal State Analysis         │
│                                 │
│ Eye Closure Duration            │
│ Yawning Duration                │
│ Feature Persistence             │
└──────────────┬──────────────────┘
               │
               ▼
┌───────────────────────┐
│ Drowsiness Decision   │
└───────────┬───────────┘
            │
       ┌────┴────┐
       ▼         ▼
    Normal     Drowsy
                 │
                 ▼
              Alert
```

---

## Lightweight Computer Vision Approach

Rather than immediately using a computationally expensive deep-learning model, NapSense focuses on lightweight computer-vision techniques.

The pipeline uses:

```
MediaPipe Face Mesh
        +
Geometric Features
        +
Texture Features
        +
Temporal Logic
```

This approach is particularly interesting for applications where:

- GPU acceleration is unavailable
- Low latency is important
- Computational resources are limited
- Deployment cost needs to remain low

The project demonstrates how carefully designed computer-vision features can form the foundation of a practical real-time AI system.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Core implementation |
| **OpenCV** | Webcam capture and image processing |
| **MediaPipe Face Mesh** | Facial landmark detection |
| **NumPy** | Numerical computation |
| **LBP** | Facial texture analysis |
| **EAR** | Eye-closure measurement |
| **MAR** | Yawning measurement |

---

## Architecture

### Component Diagram

```
                         ┌───────────────┐
                         │    Webcam     │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │    OpenCV     │
                         │ Frame Capture │
                         └───────┬───────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │ MediaPipe Face Mesh │
                      └──────────┬──────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
        Eye Landmarks      Mouth Landmarks     Face Region
              │                  │                  │
              ▼                  ▼                  ▼
             EAR                MAR                LBP
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │ Temporal Analysis   │
                      └──────────┬──────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │ Drowsiness Decision │
                      └──────────┬──────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │   Alert System      │
                      └─────────────────────┘
```

---

## Detection Logic

The conceptual decision process is:

```
                    Eye Closure
                        │
                        ▼
                 EAR < Threshold?
                        │
                        ▼
                Sustained Duration?
                        │
                        ├──────────────┐
                        │              │
                        ▼              ▼
                     Possible       Normal
                    Drowsiness      Blink
                        │
                        │
                    Yawning
                        │
                        ▼
                 MAR > Threshold?
                        │
                        ▼
                Sustained Duration?
                        │
                        ▼
                Possible Drowsiness
                        │
                        ▼
                Temporal Analysis
                        │
                        ▼
              Final Drowsiness State
                        │
                  ┌─────┴─────┐
                  ▼           ▼
                Normal      Drowsy
                              │
                              ▼
                            Alert
```

**Note:** Thresholds and temporal windows should be configurable so they can be tuned for different environments and users.

---

## Getting Started

### Prerequisites

- Python 3.9+
- Webcam
- Working microphone/speaker setup (if audio alerts are enabled)

### Installation Steps

#### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd vigil-eye
```

#### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

#### 3. Activate Virtual Environment

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows:**

```bash
.venv\Scripts\activate
```

#### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 5. Run the Application

```bash
python main.py
```

The application should initialize the webcam and begin real-time facial analysis.

**Note:** Update the commands above to match the actual repository structure if the original implementation uses a different entry point.

---

## Project Structure

```
vigil-eye/
│
├── src/
│   ├── detection/
│   │   ├── face_mesh.py
│   │   ├── eye_detection.py
│   │   ├── mouth_detection.py
│   │   └── texture_analysis.py
│   │
│   ├── features/
│   │   ├── ear.py
│   │   ├── mar.py
│   │   └── lbp.py
│   │
│   ├── monitoring/
│   │   └── temporal_analysis.py
│   │
│   ├── alerts/
│   │   └── alert_manager.py
│   │
│   └── main.py
│
├── tests/
├── requirements.txt
├── README.md
└── LICENSE
```

**Note:** Adapt this structure to the actual implementation rather than restructuring the repository unnecessarily.

---

## Evaluation

A robust evaluation of NapSense should consider both detection performance and real-time performance.

### Performance Metrics

**Detection Performance:**
- Accuracy
- Precision
- Recall
- F1 Score
- False Positive Rate
- False Negative Rate

**Real-Time Performance:**
- Detection Latency
- Frames Per Second (FPS)
- CPU Utilization

### Safety-Critical Considerations

For a safety-oriented application, particular attention should be given to:

- False negatives
- False positives
- Detection latency
- Performance under different lighting conditions

### Testing Variations

Testing should ideally include variations in:

- Lighting conditions
- Camera position
- Glasses or sunglasses
- Head orientation
- Facial structure
- Natural blinking
- Intentional yawning
- Prolonged eye closure

---

## Future Improvements

### 1. Head Pose Estimation

Add the following measurements to detect behaviors such as prolonged downward gaze:

```
Yaw
Pitch
Roll
```

### 2. Personalized Calibration

Account for individual variations in:

```
Eye shapes
Blinking frequencies
Facial geometry
Yawning patterns
```

A calibration phase could establish personalized EAR and MAR baselines.

### 3. Deep Learning Comparison

Use NapSense's lightweight approach as a baseline and compare it against:

```
MobileNet
EfficientNet
CNN-LSTM
GRU
Temporal CNN
Vision Transformer
```

The goal would be to quantify the trade-off between:

```
Accuracy
    vs.
Latency
    vs.
Computational Cost
```

### 4. Advanced Temporal Modeling

Instead of manually defined thresholds:

```
EAR Sequence
MAR Sequence
Head Pose
LBP Features
      │
      ▼
LSTM / GRU / Temporal Transformer
      │
      ▼
Drowsiness Probability
```

This could provide a more flexible representation of temporal fatigue patterns.

### 5. Multimodal Drowsiness Detection

A future version could combine visual information with other signals:

```
Facial Behavior
       +
Head Pose
       +
Voice Characteristics
       +
Driving Behavior
       +
Vehicle Telemetry
       ↓
Multimodal Drowsiness Model
```

This could make the system more robust than relying on facial information alone.

---

## Key Takeaways

NapSense reinforced an important principle:

> **AI doesn't always need to be complex to be impactful.**

A real-time safety system can be built by combining:

```
Facial Landmarks
       +
Geometric Features
       +
Texture Features
       +
Temporal Reasoning
```

The project demonstrates how classical computer vision and lightweight AI techniques can be combined to create practical, interpretable, and low-latency systems.

---

## Research Direction

The project opens up an interesting research question:

> **How effectively can lightweight facial and behavioral features detect drowsiness in real time compared with computationally heavier deep-learning approaches?**

NapSense can therefore serve as a lightweight baseline for future experiments involving temporal deep learning and multimodal AI.

---

## Limitations

**NapSense is a computer-vision prototype and should NOT be treated as a certified automotive safety system.**

Performance may be affected by:

- Poor lighting
- Camera placement
- Webcam quality
- Sunglasses
- Facial occlusion
- Extreme head angles
- Individual differences
- Landmark detection errors

Real-world automotive deployment would require extensive validation, safety testing, and appropriate certification.

---

## Author

**Vaibhav Prasad**

Computer Science & Engineering  
AI / ML / Computer Vision

---

## License

This project is available under the license specified in the repository.

---

**Last Updated:** September 2026
