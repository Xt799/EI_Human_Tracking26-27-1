# Track people, the robot and the ball with Basler cameras on Windows

Use native Windows PowerShell on the PC that can see all the cameras. Start with one fixed camera aimed at the floor. Add a second view to cover occlusions. You will get a window per camera, a shared room map, and a CSV of positions in metres. The robot heading arrow points toward its front gripper.

## 1. Install the tools

| Install on the Windows PC | Purpose and check |
| --- | --- |
| Git for Windows with Git LFS | Git checks out the code. LFS downloads the two trained model files. Check with `git --version` and `git lfs version`. [Git for Windows](https://gitforwindows.org/) · [Git LFS](https://git-lfs.com/) |
| uv | Manages Python 3.12 and the project environment. Run `winget install --id astral-sh.uv -e`, reopen PowerShell, then run `uv --version`. [uv installation](https://docs.astral.sh/uv/getting-started/installation/) |
| Basler pylon Software Suite | Install pylon Viewer and the Windows driver for your camera interface, USB3 Vision or GigE Vision. [Basler pylon](https://docs.baslerweb.com/pylon-software-suite) |
| Compatible NVIDIA graphics driver for GPU inference | The Windows lock uses CUDA 12.8 PyTorch wheels. Run `nvidia-smi` and check PyTorch below. CPU inference also works and can yield substantially fewer updates per second. |

The environment sync installs OpenCV, NumPy, SciPy, Ultralytics, PyTorch, torchvision, lap, InsightFace, ONNX Runtime and pypylon. Tracking uses the existing models. Training images, annotation software and ROS are optional for other workflows.

## 2. Get the feature and download its models

Use a native Windows checkout so uv creates a Windows virtual environment. If you need a clone, run `git clone https://github.com/Xt799/EI_Human_Tracking26-27-1.git` in PowerShell with an account that has repository access, then `cd EI_Human_Tracking26-27-1`. Run the remaining PowerShell commands from the repository root.

If the feature is published, run `git fetch origin`. If the branch exists only in your WSL checkout, transfer its feature commit using a Git bundle. In **WSL**, from that checkout, run:

```bash
git bundle create ./live-tracking.bundle feat-in-real-time-tracking ^origin/main
wslpath -w "$PWD/live-tracking.bundle"
```

In **Windows PowerShell**, inside a fresh clone, fetch the main history required by the bundle and import the branch. Paste the Windows path printed by `wslpath` when prompted:

```powershell
git fetch origin main
$bundle = Read-Host "Path to live-tracking.bundle"
git fetch "$bundle" refs/heads/feat-in-real-time-tracking:refs/heads/feat-in-real-time-tracking
```

Then select the branch and install its dependencies:

```powershell
git switch feat-in-real-time-tracking
git log -1 --oneline
git lfs install
git lfs pull --include="training/runs/detector/weights/best.pt,training/runs/heading/weights/last.pt"
uv sync --locked --python 3.12
```

Check the assets and runtime:

```powershell
Get-Item training/runs/detector/weights/best.pt, training/runs/heading/weights/last.pt, tracking/models/yolo11s-pose.pt |
    Select-Object FullName, Length
uv run --locked python -c "import cv2, torch; from pypylon import pylon; print('OpenCV', cv2.__version__, 'PyTorch', torch.__version__, 'CUDA', torch.cuda.is_available())"
```

For the models from main commit `e45c956`, the expected lengths are **5,868,997** bytes for robot/ball detection, **5,835,886** for robot heading, and **20,373,161** for person pose. A trained weight around 130 bytes is an LFS pointer. Check repository access, rerun the LFS pull and recheck its length. If CUDA is `False` on an NVIDIA machine, verify the driver and that you ran the check with `uv run` in the Windows checkout.

## 3. Verify the cameras before calibrating

1. Connect and power the first camera. In **pylon Viewer**, select it and start continuous acquisition. Move a hand through the frame to confirm that the image is updating. Check focus and exposure on both the robot and the ball at the far end of the room.
2. For multiple cameras, acquire from all of them at once in Viewer. Write down the serial number for each physical location, for example `front` and `rear`. Secure the mounts.
3. Stop acquisition and close Viewer. Clear any camera emulation setting in this PowerShell session and list the physical devices:

```powershell
Remove-Item Env:PYLON_CAMEMU -ErrorAction SilentlyContinue
uv run --locked python tracking/track.py --list-cameras
```

Expect one line per camera showing serial, model and transport. For USB3, check the cable, port, driver and available bandwidth. For GigE, check camera power and whether the camera and PC adapter have compatible IP addresses. If discovery works but opening the camera fails, close other applications that may have it open.

## 4. Calibrate the floor once for this setup

Mark four corners of a rectangle **inside the area where you want to track**. The calibrated rectangle sets the tracked area, so use as much of that area as your cameras can see. Label the physical corners A, B, C, D around the perimeter. Measure A→B (x) and A→D (y) in metres. A is the origin. Keep each physical label consistent in every view, even when the room appears reversed on screen.

The supplied lens correction applies to the original 1900×1200 lab camera/lens configuration. For different optics or geometry, supply the appropriate OpenCV camera matrix and distortion coefficients in `tracking/calibration/live/intrinsics/VIEW/` before calibration. See the [calibration guide](calibration/README.md).

In the commands below, replace `FRONT_SERIAL` and `REAR_SERIAL` with the serials from discovery and replace `4.0 3.0` with your measured dimensions. Choose the command matching your setup:

```powershell
# One camera
uv run --locked python tracking/calibrate.py --basler FRONT_SERIAL front --rect 4.0 3.0

# Two cameras sharing the same floor coordinates
uv run --locked python tracking/calibrate.py --basler FRONT_SERIAL front --basler REAR_SERIAL rear --rect 4.0 3.0
```

For each view, press **Enter** when the marks are clear, then click A, B, C, D. Use **u** to undo a click. On the final overlay, the grid should lie on the floor and the same intersection should cover the same physical spot in each view. Press **Enter** to save the profiles or **q** to cancel. If the cameras cannot see a common rectangle, the `--metres` workflow accepts six reference points per view measured from the same origin.

The profiles and grid previews are saved locally under `tracking/calibration/live/`. Calibrate all cameras that will be fused together in one session. A saved session can also run with a subset. Recalibrate after moving a camera, changing its geometry or replacing its lens.

## 5. Run the tracker

Use the same serials and view names you calibrated:

```powershell
# One view
uv run --locked python tracking/track.py --basler FRONT_SERIAL front

# Two views
uv run --locked python tracking/track.py --basler FRONT_SERIAL front --basler REAR_SERIAL rear
```

Walk across the floor with your feet visible. Move the robot between marked positions and rotate it in place. Roll the ball through the overlapping views. Check the camera overlays and top-down map. People receive persistent IDs. The owner label is optional and requires prior enrollment. Cover one camera briefly and check that objects visible in the other view continue to update.

Press **p** to pause, **c** to continue and **q** or **Esc** to stop. Add `--headless` for CSV-only operation and use **Ctrl+C** to finish. The tracker uses recent frames and drops older ones when inference falls behind. The default freshness and inter-camera arrival window is 100 ms. `--sync-ms` adjusts it. A disconnected camera that stops acquisition requires a restart to rejoin. If all inputs stall for five seconds the tracker exits with a diagnostic.

## 6. Check the results

PowerShell can open the most recent run:

```powershell
$run = Get-ChildItem results/*_positions.csv | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$rows = Import-Csv $run.FullName
$rows | Group-Object kind | Select-Object Name, Count
$rows | Where-Object kind -eq 'robot' | Select-Object -First 10 frame, t, x, y, cameras, heading_deg
```

Look for `person`, `robot` and `ball`. Compare x/y against your floor marks. The `cameras` column lists views that contributed a floor position. Robot heading uses 0° along +x and 90° along +y. A blank angle means the tracker has no current heading estimate. If boxes look right but the floor positions are wrong, check the marked corner order, measured dimensions, serial/view pairing and lens correction.

At shutdown, record fused updates/s, skipped and overwritten frames, median/p95 receipt-to-CSV latency and multi-view arrival skew. The latency clock starts when the host receives a frame. Record the GPU, camera count, resolution and synchronization setting alongside the numbers so runs are comparable.

## Test without physical cameras

In a separate PowerShell session, run the automated suite with two emulated Baslers. Expect a final `OK`. Emulation exercises concurrent acquisition and release. Controlled detections test fusion, headings, offsets and CSV values.

```powershell
$env:PYLON_CAMEMU = "2"
uv run --locked python -m unittest discover -s tests -v
Remove-Item Env:PYLON_CAMEMU
```

For a recorded-video check, a single video works without calibration. A synchronized pair can use its matching recorded floor maps and a signed frame offset. Positive offsets skip frames from the second video and negative offsets skip frames from the first. See [recorded playback](README.md#recorded-playback).
