
# 🕷️ Arachne V1

**Camera-Based Computer Vision and Motion Tracking**

Arachne V1 is a Python computer-vision prototype that transforms a webcam feed into a heads-up display (HUD). It detects motion, tracks a moving region, estimates movement, and visualises tracking information in real time.

This release focuses on the current V1 implementation.

## Features

- Live webcam feed and on-screen HUD
- Motion detection using OpenCV
- Motion-region acquisition and target locking
- Target position smoothing and short-term prediction
- Direction and speed estimation
- Movement-state classification
- Movement trail and projected velocity vector
- Tracking and prediction confidence indicators
- Approximate distance category and distance-trend estimation
- Periodic YOLO11n model inference

## Technology

- Python
- OpenCV
- Ultralytics YOLO11n
- PyTorch, used by the Ultralytics model stack

## How It Works

1. A webcam supplies video frames.
2. OpenCV compares consecutive frames to identify regions of motion.
3. A motion region is selected and tracked across frames.
4. A HUD displays the tracked region, movement estimates, trail, vector, and system status.
5. YOLO11n inference runs periodically, but its results are not yet integrated into the target-selection and tracking pipeline.

## Requirements

- Python
- A working webcam
- The dependencies listed in `requirements.txt`

## Installation

Clone or download the repository and open a terminal in the project folder.

Create a virtual environment:

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Running Arachne

Run the application from the project directory:

```powershell
python main.py
```

Arachne attempts to open the default webcam at 640 × 480 resolution.

Press **Q** while the camera window is active to exit.

The YOLO model may need to be downloaded automatically the first time it is used.

## Current Limitations

- Motion detection identifies changes between frames; it does not inherently identify what an object is.
- YOLO inference is present, but its results are not yet connected to target selection or tracking.
- Speed is estimated in pixels per frame, not real-world units.
- Distance estimates are approximate and are not calibrated physical measurements.
- Performance depends on the computer, webcam, and installed dependencies.

## Project Status

Arachne V1 is an experimental learning project exploring computer vision, motion tracking, prediction, and HUD visualisation.

It is a foundation for experimenting with intelligent visual systems, not a finished wearable or augmented-reality product.

## License

No license is currently specified. Unless a license is added, do not assume the code is available for reuse, modification, or redistribution under an open-source license.
