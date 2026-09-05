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



Move into the project directory:

```bash
cd Invisibility-Computer-Vision
```

Install the required packages:

```bash
pip install opencv-python numpy mediapipe
```

or

```bash
pip install -r requirements.txt
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
Invisibility-Computer-Vision/
│
├── main.py
├── engine.py
├── utils.py
├── requirements.txt
├── README.md
└── assets/
```

---

# Technologies Used

* Python
* OpenCV
* MediaPipe
* NumPy

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
