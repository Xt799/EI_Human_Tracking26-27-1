import csv
import json
from contextlib import ExitStack, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tracking"))

import calibrate
import live_calibration
import track
from fusion import Fusion
from sources import BaslerSource, CapturedFrame, LiveFrames, list_baslers

IMAGE = np.zeros((120, 160, 3), dtype=np.uint8)
GEOMETRY = {"Width": 160, "Height": 120}
ROOM = ((0, 4), (0, 3))
H = np.diag([0.02, 0.02, 1.0])


class Slot:
    def __init__(self):
        self.frame = None
        self.last_received = None
        self.error = None

    def offer(self, stamp):
        self.frame = CapturedFrame(IMAGE, stamp)
        self.last_received = stamp

    def take_latest(self):
        frame, self.frame = self.frame, None
        return frame


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.slots = {name: Slot() for name in ("a", "b", "c")}
        self.capture = LiveFrames(self.slots, 0.125, clock=lambda: self.now)

    def test_window_boundary_and_no_replay(self):
        for name, stamp in zip(self.slots, (10.125, 10.0, 9.999)):
            self.slots[name].offer(stamp)
        self.now = 10.125
        self.assertEqual(set(self.capture.next()), {"a", "b"})
        self.assertEqual(self.capture.discarded, {"a": 0, "b": 0, "c": 1})
        self.assertEqual(self.capture.next(), {})

    def test_slow_inference_discards_old_frames_and_keeps_latest(self):
        self.slots["a"].offer(10.1)
        self.slots["b"].offer(10.65)
        self.slots["b"].offer(10.7)
        self.now = 10.8
        self.assertEqual(self.capture.next()["b"].received_at, 10.7)
        self.slots["a"].offer(10.69)  # conversion finished after a newer view was already processed
        self.assertEqual(self.capture.next(), {})
        self.assertEqual(self.capture.discarded["a"], 2)

    def test_resume_cutoff_and_partial_then_complete_stall(self):
        self.now = 10.2
        self.slots["a"].offer(10.15)
        self.assertEqual(self.capture.next(after=10.16), {})
        self.slots["b"].offer(10.18)
        self.assertEqual(set(self.capture.next(after=10.16)), {"b"})
        self.now = 14.0
        self.slots["b"].offer(self.now)
        self.assertEqual(set(self.capture.next()), {"b"})
        self.slots["a"].error = "disconnected"
        self.now = 19.0
        with self.assertRaisesRegex(RuntimeError, "5 seconds.*a: disconnected"):
            self.capture.next()


class FusionTests(unittest.TestCase):
    def test_robot_heading_uses_confidence_and_expires_after_missing_measurements(self):
        def detection(direction, confidence):
            return (None, 1.0, None, np.zeros(2), (None, None, np.array(direction), confidence))

        robot = track.Track()
        observations = {"a": {"robot": detection([1, 0], 0.8)}, "b": {"robot": detection([0, 1], 0.6)}}
        track.fuse_objects({"robot": robot}, observations, 1.0)
        np.testing.assert_allclose(robot.heading, [0.8, 0.6])
        track.fuse_objects({"robot": robot}, {}, 1.1)
        np.testing.assert_allclose(robot.heading, [0.8, 0.6])
        track.fuse_objects({"robot": robot}, {}, 2.0)
        self.assertIsNone(robot.heading)

    def test_outlier_splits_after_persistent_evidence_with_interleaved_single_view(self):
        cam_tracks = {name: {1: track.Person()} for name in ("a", "b", "c")}
        fusion = Fusion()

        def update(t, points):
            for view, x in points.items():
                person = cam_tracks[view][1]
                person.floor_m, person.placed, person.last_seen = np.array([x, 1.0]), t, t
            return fusion.update(cam_tracks, t)

        links = update(0, {"a": 1.0, "b": 1.1, "c": 1.2})
        person = links["a", 1]
        self.assertEqual(person.placed_by, ["a", "b", "c"])
        np.testing.assert_allclose(person.floor_m, [1.1, 1.0])
        update(0.1, {"a": 1.0, "b": 1.1, "c": 3.0})
        update(0.3, {"a": 1.0})
        links = update(0.7, {"a": 1.0, "b": 1.1, "c": 3.0})
        self.assertIs(links["a", 1], person)
        self.assertIs(links["b", 1], person)
        self.assertIsNot(links["c", 1], person)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        temp = self.enterContext(TemporaryDirectory())
        self.directory = Path(temp)
        self.enterContext(patch.object(live_calibration, "LIVE_CALIB_DIR", self.directory))
        self.enterContext(patch.object(calibrate, "LIVE_CALIB_DIR", self.directory))
        for view in ("a", "b"):
            directory = self.directory / "intrinsics" / view
            directory.mkdir(parents=True)
            np.save(directory / "camera_matrix.npy", np.array([[100, 0, 80], [0, 100, 60], [0, 0, 1.0]]))
            np.save(directory / "distortion_coefficients.npy", np.zeros(5))

    def test_live_calibration_saves_shared_coordinates_and_validates_binding(self):
        pairs = [("123", "a"), ("456", "b")]
        sources = {view: SimpleNamespace(serial=serial, geometry=GEOMETRY, close=lambda: None)
                   for serial, view in pairs}
        pixels = [(0, 0), (150, 0), (150, 100), (0, 100)]
        args = SimpleNamespace(basler=pairs, cameras=["a", "b"], video=None, rect=(3, 2))
        with (patch.object(calibrate, "pylon_devices"), patch.object(calibrate, "pick_live_frame", return_value=IMAGE),
              patch.object(calibrate, "BaslerSource", side_effect=list(sources.values())),
              patch.object(calibrate, "click_points", return_value=pixels),
              patch.object(cv2, "imshow"), patch.object(cv2, "waitKey", return_value=13),
              patch.object(cv2, "destroyAllWindows"), redirect_stdout(StringIO())):
            calibrate.main(args)
        room, grid, data = live_calibration.load_live_profiles(sources)
        np.testing.assert_allclose(data["a"][0], H, atol=1e-6)
        np.testing.assert_allclose(room, [[0, 3], [0, 2]])
        self.assertEqual(grid, 0.5)
        self.assertTrue((self.directory / "floor_map_b.jpg").is_file())
        sources["a"].serial = "456"
        with self.assertRaisesRegex(ValueError, "serial or image geometry"):
            live_calibration.load_live_profiles(sources)
        sources["a"].serial = "123"
        sources["a"].geometry = GEOMETRY | {"OffsetX": 1}
        with self.assertRaisesRegex(ValueError, "serial or image geometry"):
            live_calibration.load_live_profiles(sources)
        sources["a"].geometry = GEOMETRY
        profiles = json.loads((self.directory / "profiles.json").read_text())
        profiles["b"]["room_id"] = "elsewhere"
        live_calibration.save_profiles(profiles)
        with self.assertRaisesRegex(ValueError, "share a calibration session"):
            live_calibration.load_live_profiles(sources)
        np.save(self.directory / "intrinsics" / "a" / "distortion_coefficients.npy", np.ones(5))
        with self.assertRaisesRegex(ValueError, "intrinsics have changed"):
            live_calibration.load_live_profiles({"a": sources["a"]})


class FakePose:
    def detect(self, frame):
        keypoints = np.zeros((1, 17, 2))
        keypoints[0, 15:17] = [[49, 60], [51, 60]]
        confidence = np.zeros((1, 17))
        confidence[0, 15:17] = 1
        return [7], np.array([[10, 10, 60, 60]]), keypoints, confidence


class FakeObjects:
    names = {0: "robot", 1: "ball"}

    def __call__(self, frame, **options):
        def tensor(values):
            return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array(values)))

        boxes = SimpleNamespace(xyxy=tensor([[20, 20, 40, 50], [65, 50, 75, 55]]),
                                conf=tensor([0.9, 0.9]), cls=tensor([0, 1]))
        return [SimpleNamespace(boxes=boxes)]


class FakeBasler(Slot):
    def __init__(self, serial, max_reads=None, stall_after=None):
        super().__init__()
        self.serial = serial
        self.geometry = GEOMETRY
        self.stop = SimpleNamespace(set=lambda: None)
        self.lock = ExitStack()
        self.started = self.closed = False
        self.received = self.skipped = self.overwritten = 0
        self.max_reads = max_reads
        self.stall_after = stall_after

    def start(self):
        self.started = True

    def take_latest(self):
        assert self.started
        if self.received == self.max_reads:
            raise KeyboardInterrupt
        if self.received == self.stall_after:
            self.error = "disconnected"
            return None
        self.received += 1
        self.offer(time.monotonic())
        return super().take_latest()

    def close(self):
        self.closed = True


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(TemporaryDirectory()))
        self.enterContext(patch.object(track, "RESULTS_DIR", self.directory))
        self.enterContext(patch.object(track, "PeopleDetector", FakePose))
        self.enterContext(patch.object(track, "object_detector", FakeObjects))
        self.heading = self.enterContext(patch.object(track, "RobotHeading")).return_value
        self.heading.measure.return_value = (np.array([40, 40]), np.array([40, 30]), 0.9)
        self.enterContext(patch.object(track, "lens_correction", return_value=lambda frame: frame))
        self.enterContext(patch.object(track, "load_owner", return_value=(None, None)))
        self.enterContext(patch.object(cv2, "imshow", side_effect=AssertionError("headless GUI call")))
        self.enterContext(redirect_stdout(StringIO()))

    def rows(self):
        path, = self.directory.glob("*_positions.csv")
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream))

    def assert_tracks(self, rows, views):
        self.assertEqual({r["kind"] for r in rows}, {"person", "robot", "ball"})
        self.assertEqual({r["cameras"] for r in rows}, {views})
        self.assertEqual({r["id"] for r in rows if r["kind"] == "person"}, {"P1"})
        self.assertEqual({(r["x"], r["y"]) for r in rows if r["kind"] == "person"}, {("1.000", "1.200")})
        self.assertEqual({(r["x"], r["y"]) for r in rows if r["kind"] == "robot"}, {("0.600", "0.800")})
        self.assertEqual({(r["x"], r["y"]) for r in rows if r["kind"] == "ball"}, {("1.400", "1.100")})
        self.assertEqual({r["heading_deg"] for r in rows if r["kind"] == "robot"}, {"90.0"})
        self.assertEqual({r["heading_deg"] for r in rows if r["kind"] == "ball"}, {""})
        stamps = [float(r["t"]) for r in rows]
        self.assertEqual(stamps, sorted(stamps))
        self.assertGreaterEqual(stamps[0], 0)

    def test_headless_live_cli_to_csv(self):
        for count in (1, 2, 3):
            with self.subTest(cameras=count):
                names = "abc"[:count]
                sources = [FakeBasler(str(i), 3) for i in range(count)]
                args = ["--headless"]
                for source, name in zip(sources, names):
                    args.extend(["--basler", source.serial, name])
                profiles = dict.fromkeys(names, (H, self.directory))
                with (patch.object(track, "pylon_devices"),
                      patch.object(track, "BaslerSource", side_effect=sources),
                      patch.object(track, "load_live_profiles", return_value=(ROOM, 0.5, profiles)),
                      self.assertRaises(SystemExit) as exit_status):
                    track.cli(args)
                self.assertEqual(exit_status.exception.code, 130)
                self.assertTrue(all(source.closed for source in sources))
                rows = self.rows()
                self.assertEqual(len(rows), 9)
                self.assert_tracks(rows, "+".join(names))
                self.assertEqual({r["frame"] for r in rows}, {"0", "1", "2"})
                next(self.directory.glob("*_positions.csv")).unlink()

    def test_stalled_view_does_not_supply_new_positions(self):
        sources = [FakeBasler("123", max_reads=4), FakeBasler("456", stall_after=1)]
        profiles = dict.fromkeys(("a", "b"), (H, self.directory))
        with (patch.object(track, "pylon_devices"),
              patch.object(track, "BaslerSource", side_effect=sources),
              patch.object(track, "load_live_profiles", return_value=(ROOM, 0.5, profiles)),
              self.assertRaises(SystemExit) as exit_status):
            track.cli(["--headless", "--basler", "123", "a", "--basler", "456", "b"])
        self.assertEqual(exit_status.exception.code, 130)
        rows = self.rows()
        self.assert_tracks([row for row in rows if row["frame"] == "0"], "a+b")
        self.assert_tracks([row for row in rows if row["frame"] != "0"], "a")
        self.assertEqual(len(rows), 12)

    def recording(self, name):
        path = self.directory / f"{name}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 120))
        self.assertTrue(writer.isOpened())
        for index in range(4):
            writer.write(np.full_like(IMAGE, index * 40))
        writer.release()
        return str(path)

    def test_original_two_file_cli_with_generated_videos(self):
        with (patch.object(track, "REALTIME", False), patch.object(track, "load_room", return_value=(ROOM, 0.5)),
              patch.object(track, "load_floor_map", return_value=H)):
            track.cli([self.recording("a1"), "a1", self.recording("b1"), "b1", "--headless"])
        self.assert_tracks(self.rows(), "a1+b1")
        self.assertEqual(len(self.rows()), 12)

    def test_signed_offsets_align_the_images_and_preserve_primary_frame_numbers(self):
        first, second = self.recording("first"), self.recording("second")
        for shift in (-1, 1):
            with self.subTest(offset=shift):
                values = []
                def heading(frame, box):
                    values.append(round(frame.mean() / 40))
                    return None
                self.heading.measure.side_effect = heading
                with (patch.object(track, "REALTIME", False), patch.object(track, "load_room", return_value=(ROOM, 0.5)),
                      patch.object(track, "load_floor_map", return_value=H)):
                    track.cli(["--video", first, "--position", "a4", "--video", second, "--position", "b4",
                               "--headless", "--offset", str(shift)])
                start = max(0, -shift)
                self.assertEqual(values[::2], list(range(start, start + 3)))
                self.assertEqual(values[1::2], [index + shift for index in values[::2]])
                robot = [row for row in self.rows() if row["kind"] == "robot"]
                self.assertEqual([int(row["frame"]) for row in robot], values[::2])
                self.assertTrue(all(row["heading_deg"] == "" for row in robot))
                next(self.directory.glob("*_positions.csv")).unlink()

    def test_one_uncalibrated_video_has_per_image_detections_and_no_metric_rows(self):
        with patch.object(track, "REALTIME", False):
            track.cli(["--video", self.recording("only"), "--headless"])
        self.assertEqual(self.heading.measure.call_count, 4)
        self.assertEqual(self.rows(), [])

    def test_mixed_calibration_logs_only_metric_contributors(self):
        with (patch.object(track, "REALTIME", False), patch.object(track, "load_room", return_value=(ROOM, 0.5)),
              patch.object(track, "load_floor_map", return_value=H)):
            track.cli(["--video", self.recording("one"), "--video", self.recording("two"),
                       "--position", "b4", "--headless"])
        self.assert_tracks(self.rows(), "b4")
        self.assertEqual(self.heading.measure.call_count, 8)

    def test_lfs_pointer_has_actionable_error(self):
        path = self.directory / "best.pt"
        path.write_text("version https://git-lfs.github.com/spec/v1\n")
        with self.assertRaisesRegex(ValueError, "git lfs pull"):
            track.require_trained_weight(path)

    def test_sigint_finishes_current_update_and_releases_capture(self):
        class SignallingPose(FakePose):
            def detect(self, frame):
                signal.raise_signal(signal.SIGINT)
                return super().detect(frame)

        source = FakeBasler("123")
        with (patch.object(track, "pylon_devices"),
              patch.object(track, "BaslerSource", return_value=source),
              patch.object(track, "PeopleDetector", SignallingPose),
              patch.object(track, "load_live_profiles", return_value=(ROOM, 0.5, {"a": (H, self.directory)})),
              patch.object(sys, "argv", ["track.py", "--headless", "--basler", "123", "a"]),
              self.assertRaises(SystemExit) as exit_status):
            track.entrypoint()
        self.assertEqual(exit_status.exception.code, 130)
        self.assertTrue(source.closed)
        self.assert_tracks(self.rows(), "a")
        self.assertEqual(len(self.rows()), 3)

    def test_failed_second_camera_releases_first(self):
        source = FakeBasler("123")
        with (patch.object(track, "pylon_devices"),
              patch.object(track, "BaslerSource", side_effect=[source, RuntimeError("unavailable")])):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                track.run_live([("123", "a"), ("456", "b")], True, 0.1)
        self.assertTrue(source.closed)

    def test_duplicate_serials_and_case_insensitive_views_are_rejected(self):
        for pairs in [[("123", "a"), ("123", "b")], [("123", "a"), ("456", "A")]]:
            with self.assertRaisesRegex(ValueError, "once"):
                live_calibration.validate_pairs(pairs)


@unittest.skipUnless(os.environ.get("PYLON_CAMEMU") == "2", "set PYLON_CAMEMU=2 to enable the pylon smoke test")
class BaslerEmulatorTests(unittest.TestCase):
    def test_enumerate_acquire_simultaneously_and_release(self):
        devices = [serial for serial, _, transport in list_baslers() if transport == "BaslerCamEmu"]
        self.assertEqual(len(devices), 2)
        with self.assertRaisesRegex(RuntimeError, "phantom.*unavailable"):
            BaslerSource("phantom")
        with ExitStack() as stack:
            cameras = []
            for serial in devices:
                camera = BaslerSource(serial)
                stack.callback(camera.close)
                cameras.append(camera)
            for camera in cameras:
                camera.start()
            deadline = time.monotonic() + 5
            frames = {}
            while len(frames) < 2 and time.monotonic() < deadline:
                for camera in cameras:
                    frame = camera.take_latest()
                    if frame:
                        frames[camera.serial] = frame
                    self.assertIsNone(camera.error)
                time.sleep(0.01)
            self.assertEqual(len(frames), 2)
            for frame in frames.values():
                self.assertEqual(frame.image.shape, (1200, 1900, 3))
                self.assertTrue(frame.image.flags.owndata)
        self.assertTrue(all(not camera.camera.IsOpen() and not camera.worker.is_alive() for camera in cameras))
