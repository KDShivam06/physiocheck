
PhysioCheck – AI Exercise Detection Prototype

PhysioCheck is an **AI-powered physiotherapy prototype** that uses real-time computer vision to detect exercises, validate posture, count only correct repetitions, and provide **instant visual and voice feedback** using a single camera.

---

## Overview
This prototype demonstrates how **markerless human pose estimation** can be used to guide physiotherapy exercises safely without wearables or external sensors.

---

## Features
- Real-time human pose detection (33 landmarks)
- Supported exercises:
  - Bicep Curl
  - Squats
  - Lateral Arm Raises
- Joint-angle–based posture validation
- Counts **only valid repetitions**
- Instant corrective feedback (audio + visual)
- Confidence gating for body visibility
- Fully on-device processing

---

## Tech Stack
- Python  
- OpenCV  
- MediaPipe  
- NumPy  
- pyttsx3  

---

## Requirements
- Python 3.9 – 3.11
- Webcam
- Stable lighting environment

---

## Setup

### Install Dependencies
```bash
pip install opencv-python mediapipe numpy pyttsx3
````

---

## Run

```bash
python physio.py
```

---

## Controls

```
1  → Bicep Curl
2  → Squats
3  → Arm Raise
L  → Left side
R  → Right side
Q  → Quit
```

---

## Working

1. Captures live video input from camera
2. Detects body landmarks using MediaPipe
3. Calculates joint angles and movement stages
4. Validates posture using physiotherapy rules
5. Provides real-time corrective feedback
6. Counts only correct repetitions

---

## Limitations

* No tactile feedback (pain or muscle resistance not detected)
* Accuracy depends on camera quality and lighting


---

## Disclaimer

This is a **prototype for demonstration purposes** only and is not a certified medical system.
Clinical validation is required before real-world use.

```

