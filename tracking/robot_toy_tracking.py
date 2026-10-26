#!/usr/bin/env python3
"""
robot_toy_tracking.py  --  detects the ROBOT and the TOY and sends their
floor positions to the robot over ROS 2.

Origin: EI_Human_Tracking26-27-1/cap_detection/send_pos_2.py
        Only comments were added; the code is the original.

What it does today:
  1. opens the first Basler camera, undistorts, downsizes frames to 640x480
  2. click 4 floor corners (BL, BR, TR, TL) -> homography H in CENTIMETRES
  3. every 4th frame: YOLO oriented-box model finds the robot; inside that box
     a RED and a GREEN sticker are found by HSV colour thresholds; the line
     green->red gives the robot's heading
  4. every 8th frame: a second YOLO oriented-box model finds the toy
  5. publishes robot_position (x, y, theta) and toy_pose (x, y) as Pose2D;
     toy_pose = (9999, 9999) means "toy not visible"

This script needs ROS 2 (rclpy), so it only runs in the lab / WSL. For home
development the ROS parts can be stubbed out (TODO 14). Keep it as a separate
script from people_tracking.py, or merge the detection parts into track.py
later; either way it must use the SAME floor map as the people tracker.

Changes are marked "TODO(STT) <number>:" and listed in README.md.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D
import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os
import math
from pypylon import pylon
import time

# Display dimensions
# NOTE: all detection AND the homography run on this downsized 640x480 frame,
# not on the full 1900x1200 camera image. That is fine for speed, but the
# clicked corners must come from the same downsized frame (they do, see
# get_roi_points), and a floor map made by calibrate.py on a full-size frame
# cannot be reused here without rescaling. TODO(STT) 12 covers this.
DISPLAY_W, DISPLAY_H = 640, 480
# MIN_MARKER_AREA = 3  # pixels
MIN_MARKER_AREA = 5  # pixels
# diff camera
# NOTE: defined but never called (main() opens the first camera found).
# Harmless; can be deleted.
def open_camera_by_serial(target_serial: str):
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()

    if not devices:
        raise RuntimeError("No Basler cameras detected.")

    for dev in devices:
        if dev.GetSerialNumber() == target_serial:
            cam = pylon.InstantCamera(factory.CreateDevice(dev))
            cam.Open()
            return cam

    raise RuntimeError(f"Camera with serial {target_serial} not found.")

class ObjectPositionPublisher:
    def __init__(self, node: Node):
        self.node = node
        self.robot_pub = node.create_publisher(Pose2D, 'robot_position', 10)
        self.toy_pub = node.create_publisher(Pose2D, 'toy_pose', 10)

    def publish_robot(self, x, y, theta):
        msg = Pose2D(x=float(x), y=float(y), theta=float(theta))
        self.robot_pub.publish(msg)
        self.node.get_logger().info(f"[Robot] x={x:.2f} m, y={y:.2f} m, theta={theta:.3f} rad")

    def publish_target_toy(self, x, y):
        msg = Pose2D(x=float(x), y=float(y), theta=0.0)
        self.toy_pub.publish(msg)
        self.node.get_logger().info(f"[Target Toy] x={x:.2f} m, y={y:.2f} m")


# TODO(STT) 12: second copy of the click-the-corners calibration. Click order
# here is BL, BR, TR, TL with origin at BL; people_tracking.py uses TL, TR, BL,
# BR with origin at TL. Delete this function and load the floor map saved by
# calibrate.py instead, so robot, toy and people share one coordinate system.
def get_roi_points(frame):
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(window_name, frame)

    window_name = "Select 4 ROI points (BL -> BR -> TR -> TL)"
    while True:
        display_frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
        cv2.imshow(window_name, display_frame)
        cv2.setMouseCallback(window_name, click_event)
        if len(points) == 4:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    return np.array(points, dtype=np.float32)


# color mask generation
# TODO(STT) 13: robot HEADING comes from a red and a green sticker on the robot,
# found with these fixed HSV ranges. HSV ranges are lighting-dependent: they
# were tuned for last year's room. In our room either
#   (a) retune the six numbers below on our footage (quick), or
#   (b) replace the stickers with a printed ArUco marker on the robot and use
#       cv2.aruco.ArucoDetector: it gives position AND heading, works in any
#       light, and removes this function, find_largest_centroid() and the
#       sticker block in main() entirely (recommended).
def make_color_masks(hsv):
    # color ranges
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_green = np.array([40, 60, 60])
    upper_green = np.array([90, 255, 255])

    # Masks
    r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red = cv2.bitwise_or(r1, r2)
    green = cv2.inRange(hsv, lower_green, upper_green)

    # Simplified morphology
    kernel = np.ones((3, 3), np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)

    return red, green


# centroid finding
def find_largest_centroid(mask_img, min_area=MIN_MARKER_AREA):
    # Find contours and select the most circular largest area
    cnts, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    best_cnt = None
    best_score = -1
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter ** 2)
        score = circularity * area  # prioritize circular and larger contours
        if score > best_score:
            best_score = score
            best_cnt = c

    if best_cnt is None:
        return None

    M = cv2.moments(best_cnt)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)


def main():
    # TODO(STT) 14: the three ROS lines here, the ObjectPositionPublisher class
    # above and the rclpy.ok()/destroy_node()/shutdown() lines at the end are the
    # only ROS parts. To run this at home without ROS, wrap them in
    # "if USE_ROS:" and print the positions instead of publishing.
    rclpy.init()
    node = rclpy.create_node('toys_following_node')
    publisher = ObjectPositionPublisher(node)

    # TODO(STT) 11: previous student's Linux home folder. Point at this repo:
    #   package_path = os.path.dirname(os.path.abspath(__file__))
    # and change the sub-folder names below from 'models'/'media' to
    # 'models'/'calibration'.
    package_path = '/home/haichen/ros2_ws/src/cap_detection/cap_detection/cap_Detection/my_robot_pkg_'

    # these work best so far
    # robot_model = YOLO(os.path.join(package_path, 'models', 'best_robot_test.pt'))
    # toy_model = YOLO(os.path.join(package_path, 'models', 'best (4).pt'))
    # robot_model = YOLO(os.path.join(package_path, 'models', 'robot_best_n.pt'))

    # TODO(STT) 11: 'robot_test.pt' is not included in any inherited repository.
    # Obtain it from the previous group or train an OBB model on our recordings
    # (README: "Training a detector"). Until then the robot part cannot run.
    robot_model = YOLO(os.path.join(package_path, 'models', 'robot_test.pt'))
    # TODO(STT) 11: 'best (4).pt' is models/toy_best.pt in this repository:
    # YOLO11m-OBB, one class "toy", trained at imgsz 1024. It is an OBB model,
    # so reading result.obb below is correct. Change the path.
    toy_model = YOLO(os.path.join(package_path, 'models', 'best (4).pt'))

    camera_matrix = np.load(os.path.join(package_path, 'media', 'camera_matrix.npy'))
    dist_coeffs = np.load(os.path.join(package_path, 'media', 'distortion_coefficients.npy'))

    # TODO(STT) 2: same as in people_tracking.py: for home development read an
    # MP4 with cv2.VideoCapture instead of the Basler camera (here and in the
    # RetrieveResult lines in the loop).
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    try:
        camera.AcquisitionFrameRateEnable.SetValue(True)
        camera.AcquisitionFrameRate.SetValue(10.0)
    except Exception:
        node.get_logger().warn("Could not set custom frame rate")

    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    # Get initial frame for ROI selection
    frame = None
    while camera.IsGrabbing():
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grab_result.GrabSucceeded():
            image = converter.Convert(grab_result)
            frame = image.GetArray()
            grab_result.Release()
            break
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    frame_display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
    # TODO(STT) 12: floor rectangle hard-coded as 360 x 225 CENTIMETRES (last
    # year's room), and positions are divided by 100 further down to get metres.
    # Replace these 3 lines with  H = np.load("calibration/floor_map_<view>.npy")
    # from calibrate.py (metres, same corner order as people_tracking.py) and
    # remove every "/ 100.0" below.
    roi_pixel_points = get_roi_points(frame_display)
    roi_real_points = np.array([[0, 0], [360, 0], [360, 225], [0, 225]], dtype=np.float32)
    H, _ = cv2.findHomography(roi_pixel_points, roi_real_points)

    # Tracking variables
    last_red_center = None
    last_green_center = None
    last_robot_pos = None

    # Performance optimization variables
    frame_count = 0
    process_every_n_frame = 4  # Process every other frame
    toy_process_every_n_frame = 8  # Process toys less frequently

    # For FPS calculation
    fps = 0
    fps_update_time = time.time()
    fps_frame_count = 0

    while rclpy.ok() and camera.IsGrabbing():
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grab_result.GrabSucceeded():
            image = converter.Convert(grab_result)
            frame = image.GetArray()
            grab_result.Release()

        frame_count += 1

        # Skipping frames for performance
        if frame_count % process_every_n_frame != 0:
            continue

        # Start timing
        start_time = time.time()

        # Preprocess frame
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        frame_display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

        # Convert to HSV for color detection
        hsv = cv2.cvtColor(frame_display, cv2.COLOR_BGR2HSV)

        # Detect robot using YOLO
        # TODO(STT) 13: if the ArUco option is chosen, this whole robot block
        # (down to "Detect toys") is replaced by ~15 lines of cv2.aruco code:
        # detect marker -> its 4 corners -> centre through H = position,
        # corner0->corner1 direction through H = heading. No YOLO robot model
        # needed at all, which also solves the missing robot_test.pt (TODO 11).
        results = robot_model.predict(
            frame_display,
            task='obb',
            imgsz=320,  # Reduced resolution
            conf=0.60,  # it was 0.8  
            iou=0.45,
            verbose=False,
            agnostic_nms=True,
            max_det=1  # Only detect one robot
        )

        robot_detected = False
        red_center = None
        green_center = None

        # works good 
        for result in results:
            if result.obb is None:
                continue

            for i in range(len(result.obb.cls)):
                class_id = int(result.obb.cls[i].item())
                if class_id != 0:  # Robot class
                    continue

                robot_detected = True
                cx, cy, w, h, angle = result.obb.xywhr[i].cpu().numpy()
                rect = ((cx, cy), (w, h), -np.degrees(angle))
                box = cv2.boxPoints(rect).astype(np.int32)
                cv2.drawContours(frame_display, [box], 0, (0, 255, 0), 2)

                real = cv2.perspectiveTransform(np.array([[[cx, cy]]], dtype=np.float32), H)
                robot_x, robot_y = real[0][0] / 100.0

                # Calculate expanded bounding box with padding
                padding_factor = 0.1
                x_min = max(0, int(min(box[:, 0]) - padding_factor * w))
                y_min = max(0, int(min(box[:, 1]) - padding_factor * h))
                x_max = min(DISPLAY_W, int(max(box[:, 0]) + padding_factor * w))
                y_max = min(DISPLAY_H, int(max(box[:, 1]) + padding_factor * h))

                # Create mask for robot region
                mask = np.zeros((DISPLAY_H, DISPLAY_W), dtype=np.uint8)
                cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)

                # Apply mask to HSV image
                roi_masked = cv2.bitwise_and(hsv, hsv, mask=mask)

                # Detect stickers in the robot region
                mask_red_padded, mask_green_padded = make_color_masks(roi_masked)
                red_center_padded = find_largest_centroid(mask_red_padded)
                green_center_padded = find_largest_centroid(mask_green_padded)

                cv2.rectangle(frame_display, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

                # Use stickers found in robot region if available
                if red_center_padded and green_center_padded:
                    red_center = red_center_padded
                    green_center = green_center_padded

        if red_center and green_center:
            last_red_center = red_center
            last_green_center = green_center

            cv2.line(frame_display, green_center, red_center, (255, 0, 0), 2)
            cv2.circle(frame_display, red_center, 8, (0, 0, 255), -1)
            cv2.circle(frame_display, green_center, 8, (0, 255, 0), -1)

            green_real = cv2.perspectiveTransform(
                np.array([[[green_center[0], green_center[1]]]], dtype=np.float32), H
            )[0][0]

            red_real = cv2.perspectiveTransform(
                np.array([[[red_center[0], red_center[1]]]], dtype=np.float32), H
            )[0][0]

            dx = red_real[0] - green_real[0]
            dy = red_real[1] - green_real[1]
            theta = math.atan2(dy, dx)

            publisher.publish_robot(robot_x, robot_y, theta)
            last_robot_pos = (robot_x, robot_y, theta)

            deg = math.degrees(theta)
            cv2.putText(frame_display, f"Angle: {deg:.1f}deg",
                        (green_center[0] + 10, green_center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame_display, f"Pos: ({robot_x:.2f},{robot_y:.2f})m",
                        (green_center[0] + 10, green_center[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        elif last_robot_pos:
            # Use last known position if current detection fails
            robot_x, robot_y, theta = last_robot_pos
            publisher.publish_robot(robot_x, robot_y, theta)
            cv2.putText(frame_display, "Using last known position", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame_display, "Stickers not found", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Detect toys
        if frame_count % toy_process_every_n_frame == 0:
            toy_results = toy_model.predict(
                frame_display,
                task='obb',
                # imgsz=320, # then 320 does not always detect the toy  
                imgsz=640,
                conf=0.8,
                verbose=False,
                max_det=1
            )
            toy_detected = False
            for result in toy_results:
                if result.obb is None:
                    continue
                for i in range(len(result.obb.cls)):
                    conf = result.obb.conf[i].item()
                    if conf > 0.5:
                        cx, cy, w, h, angle = result.obb.xywhr[i].cpu().numpy()
                        real = cv2.perspectiveTransform(np.array([[[cx, cy]]], dtype=np.float32), H)
                        toy_x, toy_y = real[0][0] / 100.0
                        publisher.publish_target_toy(toy_x, toy_y)
                        toy_detected = True

                        # Draw toy
                        rect = ((cx, cy), (w, h), -np.degrees(angle))
                        box = cv2.boxPoints(rect).astype(np.int32)
                        cv2.drawContours(frame_display, [box], 0, (0, 0, 255), 2)
                        cv2.putText(frame_display, f"Toy ({toy_x:.2f},{toy_y:.2f})",
                                    (int(cx) + 10, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        break
            if not toy_detected:
                # indicate that the toy is not found
                # KEEP: (9999, 9999) is the agreed "toy not visible" signal the
                # robot-side controller (move.py, see README) checks for.
                publisher.publish_target_toy(9999, 9999)
                cv2.putText(frame_display, "Toy not found", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Calculate and display FPS
        fps_frame_count += 1
        if time.time() - fps_update_time >= 1.0:
            fps = fps_frame_count / (time.time() - fps_update_time)
            fps_update_time = time.time()
            fps_frame_count = 0

        processing_time = time.time() - start_time
        cv2.putText(frame_display, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_display, f"Proc: {processing_time * 1000:.1f}ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Tracking", frame_display)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('p'):
            while cv2.waitKey(0) not in [ord('c'), 27]:
                pass

        # If you ever add subscriptions/timers, you can process callbacks:
        # rclpy.spin_once(node, timeout_sec=0.0)

    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
