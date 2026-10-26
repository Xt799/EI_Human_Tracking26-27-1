#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Basler camera video recorder.

Origin: uploaded by a group member to the project folder as basler_record_video.py.
Only comments were added.

Purpose for us: run this IN THE LAB, with the real cameras, to record the
footage that everyone develops against at home (people_tracking.py reads an
MP4 instead of the camera, see TODO(STT) 2 there).

The recordings of 12 Oct 2026 (see README "Recordings") were made with this
script. If another session ever happens, record:
  - a few seconds with nobody in the room, floor clearly visible, camera not
    moved afterwards (calibration frame)
  - the owner walking, sitting, bending, picking something up
  - a second person (stranger) doing the same, then both together, crossing paths
  - the robot and the toy on the floor

Stop recording:
    - Press SPACE
    - Or press q
    - Or press Ctrl+C in terminal
"""

# -------- VM-friendly GUI env: must be before importing cv2 --------
import os
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["QT_X11_NO_MITSHM"] = "1"
os.environ["QT_OPENGL"] = "software"

import time
from datetime import datetime

import cv2
from pypylon import pylon


# ========================= CONFIG =========================

CAM_WIDTH = 1900
CAM_HEIGHT = 1200

# This is the video file FPS.
# It should match the real camera FPS for real-time playback.
RECORD_FPS = 30.0

# TODO(STT) 15: previous student's Linux path. Change to a folder next to this
# repo, e.g.  OUTPUT_DIR = "../recordings"  (recordings/ is in .gitignore so
# large videos are never committed by accident).
OUTPUT_DIR = "/home/haichen/ros2_ws/src/Video_from_basler"

# Keep MP4
FOURCC = "mp4v"
OUTPUT_EXT = ".mp4"

SHOW_PREVIEW = True
DISPLAY_SCALE = 0.50

WINDOW_NAME = "Basler Recording - Press SPACE to stop"


# ========================= FUNCTIONS =========================

def open_first_basler_camera():
    """
    Open the first Basler camera detected by the laptop.
    No serial number is required.
    """
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()

    if not devices:
        raise RuntimeError(
            "No Basler camera detected. Check USB cable, power, or pylon driver."
        )

    print("Detected Basler cameras:")
    for i, dev in enumerate(devices):
        print(
            f"  [{i}] Serial: {dev.GetSerialNumber()} | "
            f"Name: {dev.GetFriendlyName()}"
        )

    print("Using the first detected Basler camera.")
    camera = pylon.InstantCamera(factory.CreateDevice(devices[0]))
    camera.Open()

    return camera


def set_camera_fps_if_possible(camera, fps):
    """
    Try to make the Basler camera FPS match the saved video FPS.
    This is not speeding up or slowing down the video.
    It just helps the saved video play in real time.
    """
    try:
        # Some Basler cameras use AcquisitionFrameRateEnable
        if hasattr(camera, "AcquisitionFrameRateEnable"):
            camera.AcquisitionFrameRateEnable.SetValue(True)

        # Most newer Basler cameras use AcquisitionFrameRate
        if hasattr(camera, "AcquisitionFrameRate"):
            camera.AcquisitionFrameRate.SetValue(float(fps))
            print(f"Camera FPS set to {fps}")
            return

        # Some older models use AcquisitionFrameRateAbs
        if hasattr(camera, "AcquisitionFrameRateAbs"):
            camera.AcquisitionFrameRateAbs.SetValue(float(fps))
            print(f"Camera FPS set to {fps}")
            return

        print("Camera FPS setting not available. Continue with default camera FPS.")

    except Exception as e:
        print(f"Could not set camera FPS: {e}")
        print("Continue with default camera FPS.")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        OUTPUT_DIR,
        f"basler_record_{timestamp}{OUTPUT_EXT}"
    )

    camera = None
    writer = None

    try:
        # -------- Open first detected Basler camera --------
        camera = open_first_basler_camera()

        # -------- Set resolution --------
        try:
            camera.Width.SetValue(CAM_WIDTH)
            camera.Height.SetValue(CAM_HEIGHT)
        except Exception as e:
            print(f"Resolution set warning: {e}")
            print("Continue with camera default resolution.")

        # -------- Try to match camera FPS with video file FPS --------
        set_camera_fps_if_possible(camera, RECORD_FPS)

        width = int(camera.Width.GetValue())
        height = int(camera.Height.GetValue())

        print(f"Recording resolution: {width} x {height}")
        print(f"Video file FPS: {RECORD_FPS}")

        # -------- Convert Basler image to OpenCV BGR image --------
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        # -------- Create video writer --------
        fourcc = cv2.VideoWriter_fourcc(*FOURCC)
        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            RECORD_FPS,
            (width, height)
        )

        if not writer.isOpened():
            raise RuntimeError(
                "VideoWriter failed to open. MP4 writer is not available."
            )

        # -------- Start camera grabbing --------
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        # -------- Preview window --------
        if SHOW_PREVIEW:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(
                WINDOW_NAME,
                int(width * DISPLAY_SCALE),
                int(height * DISPLAY_SCALE)
            )

        print("Recording started.")
        print(f"Saving to: {output_path}")
        print("Press SPACE to stop recording.")
        print("You can also press q or Ctrl+C.")

        frame_count = 0
        start_time = time.time()

        last_print_time = start_time
        recent_frame_count = 0

        while camera.IsGrabbing():
            grab = camera.RetrieveResult(
                5000,
                pylon.TimeoutHandling_ThrowException
            )

            if grab.GrabSucceeded():
                frame = converter.Convert(grab).GetArray()

                # Save original full-resolution frame
                writer.write(frame)

                frame_count += 1
                recent_frame_count += 1

                # Show preview
                if SHOW_PREVIEW:
                    preview = cv2.resize(
                        frame,
                        (0, 0),
                        fx=DISPLAY_SCALE,
                        fy=DISPLAY_SCALE
                    )

                    cv2.putText(
                        preview,
                        "Recording... Press SPACE to stop",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2
                    )

                    cv2.imshow(WINDOW_NAME, preview)

            grab.Release()

            # Print actual recording FPS every second
            now = time.time()
            if now - last_print_time >= 1.0:
                actual_fps_now = recent_frame_count / (now - last_print_time)
                print(f"Current actual grabbing FPS: {actual_fps_now:.2f}")
                recent_frame_count = 0
                last_print_time = now

            # Keyboard check
            key = cv2.waitKey(1) & 0xFF

            # SPACE key
            if key == 32:
                print("SPACE pressed. Stop recording.")
                break

            # q key
            if key == ord("q"):
                print("q pressed. Stop recording.")
                break

        elapsed = time.time() - start_time
        actual_fps = frame_count / elapsed if elapsed > 0 else 0.0

        print("Recording finished.")
        print(f"Saved video: {output_path}")
        print(f"Frames saved: {frame_count}")
        print(f"Elapsed real time: {elapsed:.2f} seconds")
        print(f"Actual grabbing FPS: {actual_fps:.2f}")
        print(f"Video file FPS: {RECORD_FPS}")

        if abs(actual_fps - RECORD_FPS) <= 3:
            print("FPS is close enough. Playback should look like real time.")
        else:
            print("Warning: actual FPS and video FPS are different.")
            print("If playback speed still looks wrong, set RECORD_FPS close to Actual grabbing FPS.")

    except KeyboardInterrupt:
        print("\nCtrl+C pressed. Stop recording safely.")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # -------- Release video writer --------
        if writer is not None:
            writer.release()

        # -------- Close camera --------
        if camera is not None:
            try:
                if camera.IsGrabbing():
                    camera.StopGrabbing()
                if camera.IsOpen():
                    camera.Close()
            except Exception:
                pass

        # -------- Close OpenCV window --------
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print("Clean exit.")


if __name__ == "__main__":
    main()