# Ghost / Invisibility Mode

A real-time computer vision application that creates an invisibility effect using **Python**, **OpenCV**, and **MediaPipe**. The project performs real-time human segmentation and replaces the detected person with a previously captured background, creating the illusion of becoming invisible.

---

# Features

* Real-time invisibility effect
* Background calibration
* MediaPipe Selfie Segmentation
* Hand gesture controls
* Live webcam processing
* Screenshot capture
* Background recalibration
* Multi-camera support
* Lightweight and easy to run

---

# How It Works

The application begins by capturing a clean background frame during initialization.

Using **MediaPipe Selfie Segmentation**, the system detects the user's body in every video frame. When invisibility mode is activated, the segmented foreground is replaced with the stored background frame, making the person appear invisible.

Hand gesture recognition runs simultaneously using **MediaPipe Hands**.

Supported gestures:

* Spread both hands apart to display the interaction area.
* Pinch the thumb and index finger together on either hand to toggle invisibility.
* Pinch again to return to normal.

---

# Requirements

* Python **3.10**, **3.11**, or **3.12**
* Webcam
* Windows, Linux, or macOS

> **Recommended:** Python 3.10 for the best compatibility.

---

# Installation

1. Clone the repository:
```bash
git clone https://github.com/Harnoor001/The-Ghost-Project.git
cd The-Ghost-Project
```

2. Create and activate a virtual environment:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the project and its dependencies:
```bash
pip install .
```

---

# Running the Project

Run using the default webcam:

```bash
python main.py
```

Use another camera:

```bash
python main.py 1
```

---

# Controls

| Key     | Function               |
| ------- | ---------------------- |
| **R**   | Recalibrate background |
| **S**   | Save screenshot        |
| **Q**   | Quit                   |
| **ESC** | Quit                   |

---

# Gesture Controls

### Calibration

Remain still for approximately **3 seconds** while the application captures the background.

### Show Interaction Box

Spread both hands apart until a yellow rectangle appears.

### Become Invisible

Pinch your thumb and index finger together on either hand.

### Become Visible Again

Repeat the pinch gesture.

---

# Project Structure

```
The-Ghost-Project/
│
├── pyproject.toml
├── main.py
├── core/
├── vision/
├── interaction/
├── effects/
├── rendering/
└── README.md
```

---

# Technologies Used

* Python
* OpenCV
* MediaPipe
* NumPy

---

## Background Model 2.0

The background model uses a streaming, memory-efficient design to capture and maintain the reference background frame used for the invisibility effect.

### Streaming Calibration

During the 3-second calibration phase the model accumulates frames using **per-pixel weighted averaging**.

- **First 15 frames:** all pixels are accepted without a foreground mask, giving the segmentation model time to stabilise.
- **Subsequent frames:** a foreground mask from MediaPipe Selfie Segmentation is applied. Each pixel accumulates only its *background* contribution, weighted by `1.0 − foreground_mask`. This means a pixel that is partially or fully occupied by a person is excluded (or down-weighted) from the background estimate, preventing the person from contaminating the captured background.

**Why per-pixel counts matter:**

| Pixel | Frame 1 | Frame 2 | Frame 3 | Naïve mean | Correct mean |
|-------|---------|---------|---------|-----------|-------------|
| Wall  | 10      | 20      | 30      | 20        | **20**      |
| Person (foreground) | 100 | 100 | 100 | 66.7 (wrong) | **fallback ≈ 20** (correct) |

Using a global frame count for masked pixels produces artificially dark values. The per-pixel count eliminates this error.

### Foreground Protection During EMA

After calibration, the model optionally refines the background in real time using **Exponential Moving Average (EMA)**:

```
background ← (1 − eff_alpha) × background + eff_alpha × frame
eff_alpha[h,w] = update_rate × (1 − foreground_mask[h,w])
```

With `update_rate = 0.02`:
- **Background pixels:** normal slow update (α ≈ 0.02).
- **Person pixels (mask ≈ 1):** eff_alpha ≈ 0, effectively no update — the person is never absorbed into the background.
- **Uncertain pixels (mask ≈ 0.5):** partial update.

### Gradual Lighting Adaptation

The EMA slowly adapts to lighting changes. At 30 FPS with α = 0.02, small room-lighting changes will gradually propagate into the background over several seconds. Sudden, large changes (e.g., switching lights off) take longer — **press R to recalibrate** for fast recovery.

### Stale Detection

If the model goes more than **5 seconds** without receiving a meaningful background update (i.e., every frame has been fully foreground), it is marked **stale**.

- The HUD displays: *"Background stale — press R to recalibrate"*
- The invisibility effect is automatically disabled until the background is valid again.
- The stale background data is **not destroyed** — it is simply not fed to the effect.

### Recalibration

Press **R** at any time to:
1. Reset the background model.
2. Run a fresh 3-second calibration.
3. Resume normal operation.

### Limitations

- **Camera movement:** if the camera is physically moved, the captured background becomes incorrect. The model has no camera-motion compensation. Press R to recalibrate after repositioning the camera.
- **Multi-person scenes:** background protection is only as accurate as the segmentation mask. MediaPipe Selfie Segmentation is optimised for a single person; results with multiple people may be imperfect.
- **Memory:** the model uses O(frame\_size) memory — three arrays of shape H×W×3 or H×W at most. No frame history is stored.

---

# Troubleshooting

## Error

```
AttributeError: module 'mediapipe' has no attribute 'solutions'
```

This usually indicates one of the following:

* An incompatible MediaPipe installation.
* A corrupted MediaPipe installation.
* A file named `mediapipe.py` or folder named `mediapipe` exists inside the project.
* An unsupported Python version is being used.

---

## Step 1 – Check MediaPipe Version

Run:

```bash
py -m pip show mediapipe
```

Compatible versions include:

```
0.10.5
0.10.7
0.10.9
0.10.11
0.10.14
0.10.21
```

---

## Step 2 – Reinstall MediaPipe

Remove the existing installation:

```bash
py -m pip uninstall mediapipe
```

Install a compatible version:

```bash
py -m pip install mediapipe==0.10.21
```

---

## Step 3 – Verify Installation

Run Python:

```bash
py
```

Execute:

```python
import mediapipe as mp

print(mp.__version__)
print(hasattr(mp, "solutions"))
```

Expected output:

```
0.10.21
True
```

---

## Step 4 – Check for Naming Conflicts

Ensure your project does **not** contain:

```
mediapipe.py
```

or

```
mediapipe/
```

Rename or remove them if present.

---

## Step 5 – Check Python Version

Run:

```bash
py --version
```

For best compatibility, use:

```
Python 3.10
Python 3.11
Python 3.12
```

If using Python 3.14 or later and compatibility issues occur, install Python 3.10–3.12 and reinstall the project dependencies.

---

# Screenshots

Add screenshots or GIF demonstrations here.

Example:

```
screenshots/
├── calibration.png
├── invisible-mode.png
└── gesture-control.gif
```

---

# Future Improvements

* Multiple invisibility modes
* Background blur mode
* Green screen replacement
* Virtual backgrounds
* Object invisibility
* Performance optimization using GPU acceleration
* Gesture customization

---

# Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

---

License

This project is licensed under the MIT License.
