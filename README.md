# Ghost / Invisibility Mode

Ghost is a local desktop computer-vision application that replaces a person in a webcam stream with a calibrated background. It uses OpenCV for capture/rendering and MediaPipe for person segmentation and hand tracking. No frames are uploaded or stored by the application unless you explicitly save a screenshot.

## Production readiness

- Reproducible Python package with a `ghost` command
- Supported Python versions: 3.10, 3.11, and 3.12
- Pinned MediaPipe runtime compatible with the legacy `mp.solutions` API
- Graceful errors for missing packages, camera startup failure, interrupted runs, and camera-frame loss
- Unit tests for the background model and gesture state machine
- CI checks for supported Python versions

## Requirements

- Python 3.10–3.12
- A webcam accessible to the current user
- Windows, macOS, or Linux with a desktop session (OpenCV needs a display)

## Install

Clone the repository, then create an isolated environment.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
```

On macOS/Linux, activate with `source .venv/bin/activate`.

For development and release checks:

```powershell
python -m pip install -e ".[dev]"
```

## Run

```powershell
ghost
```

Or run directly from a checkout:

```powershell
python main.py
```

Use another camera or adjust the capture settings when needed:

```powershell
ghost --camera 1 --width 1280 --height 720 --fps 30
ghost --help
```

The legacy positional camera form (`ghost 1`) is also supported. Screenshots are saved in `screenshots/` by default; change this with `--screenshot-dir PATH`.

## Use

1. Start the app and keep the camera fixed.
2. **Keep the scene clear during the three-second calibration.** The app captures the reference background during this period.
3. Spread both hands to show the visual interaction box.
4. Pinch your thumb and index finger on either hand to toggle invisibility.

| Input | Action |
| --- | --- |
| `R` | Recalibrate the background |
| `S` | Save a PNG screenshot |
| `Q` or `Esc` | Quit |
| Pinch | Toggle visibility |

## Operational notes

- Keep the camera fixed after calibration. If the camera moves or lighting changes dramatically, press `R`.
- The background model adapts gradually to normal lighting changes while avoiding updates in detected foreground regions.
- If the scene changes substantially, press `R` to recalibrate. A temporary lack of background pixels while you are in frame does not disable invisibility.
- Close apps such as Teams, Zoom, or browser tabs that are using the webcam before launching Ghost.
- The app requires a graphical desktop and is not intended for headless servers.

## Verification and release build

Run the full local gate before a release:

```powershell
python -m pytest
python -m ruff check .
python -m build
```

The generated wheel and source archive are placed in `dist/`. A physical webcam smoke test remains required because CI cannot validate local camera permissions, drivers, or lighting.

## Project layout

```text
core/          camera, configuration, and per-frame state
vision/        segmentation, hand tracking, and background model
interaction/   gesture state machine
effects/       invisibility compositing effect
rendering/     HUD, hand mesh, and portal visuals
tests/         hardware-independent unit tests
```

## Privacy

Processing happens locally in memory. The application does not send webcam frames over the network and only writes image data when you press `S`.

## Security and support

Report security issues privately; see [SECURITY.md](SECURITY.md). General project contributions should include tests for behavior changes.

## License

MIT. See [LICENSE.txt](LICENSE.txt).
