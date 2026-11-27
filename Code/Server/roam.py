import os
import sys
import time
from typing import Optional
import cv2
import numpy as np
from motor import Ordinary_Car
from ultrasonic import Ultrasonic
from servo import Servo

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class RoamCar:
    """
    High-level abstraction for controlling the Freenove 4WD Smart Car.
    Provides movement primitives (run, wait, go_around_left, go_around_right,
    turn_around) and autonomous free-roaming behavior.
    """

    def __init__(
        self,
        default_speed: int = 1000,
        turn_speed: int = 1500,
        turn_around_duration: float = 1.2,
        wall_threshold_cm: float = 30.0,
    ):
        self.default_speed = default_speed
        self.turn_speed = turn_speed
        self.turn_around_duration = turn_around_duration
        self.wall_threshold_cm = wall_threshold_cm

        self.motor = Ordinary_Car()
        self.sonic = Ultrasonic()
        self.servo = Servo()

        # Center the ultrasonic sensor servo to face straight ahead (90 degrees)
        self.center_servo()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()

    def center_servo(self) -> None:
        """Point the pan/tilt servos straight ahead."""
        self.servo.set_servo_pwm('0', 90)
        self.servo.set_servo_pwm('1', 90)

    # -------------------------------------------------------------------------
    # Movement Abstractions
    # -------------------------------------------------------------------------

    def run(
        self,
        speed: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> None:
        """
        Drive straight forward.
        If duration is provided, drives for `duration` seconds and then stops.
        """
        spd = self.default_speed if speed is None else speed
        self.motor.set_motor_model(spd, spd, spd, spd)
        if duration is not None:
            time.sleep(duration)
            self.wait()

    def backward(
        self,
        speed: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> None:
        """
        Drive straight backward.
        If duration is provided, drives backward for `duration` seconds and stops.
        """
        spd = abs(self.default_speed if speed is None else speed)
        self.motor.set_motor_model(-spd, -spd, -spd, -spd)
        if duration is not None:
            time.sleep(duration)
            self.wait()

    def wait(self, duration: Optional[float] = None) -> None:
        """
        Stop all motors immediately.
        If duration is provided, remains stopped for `duration` seconds.
        """
        self.motor.set_motor_model(0, 0, 0, 0)
        if duration is not None:
            time.sleep(duration)

    def go_around_left(
        self,
        outer_speed: int = 2000,
        inner_speed: int = 600,
        duration: Optional[float] = None,
    ) -> None:
        """
        Arc/curve to the left by driving the right (outer) wheels faster
        than the left (inner) wheels.
        If duration is provided, curves for `duration` seconds and then stops.
        """
        self.motor.set_motor_model(inner_speed, inner_speed, outer_speed, outer_speed)
        if duration is not None:
            time.sleep(duration)
            self.wait()

    def go_around_right(
        self,
        outer_speed: int = 2000,
        inner_speed: int = 600,
        duration: Optional[float] = None,
    ) -> None:
        """
        Arc/curve to the right by driving the left (outer) wheels faster
        than the right (inner) wheels.
        If duration is provided, curves for `duration` seconds and then stops.
        """
        self.motor.set_motor_model(outer_speed, outer_speed, inner_speed, inner_speed)
        if duration is not None:
            time.sleep(duration)
            self.wait()

    def turn_around(
        self,
        speed: Optional[int] = None,
        duration: Optional[float] = None,
        direction: str = "right",
    ) -> None:
        """
        Pivot 180 degrees in place to face the opposite direction.
        Adjust `duration` or `turn_around_duration` to calibrate a clean 180° turn
        on your floor surface.
        """
        spd = self.turn_speed if speed is None else speed
        dur = self.turn_around_duration if duration is None else duration

        if direction.lower() == "left":
            self.motor.set_motor_model(-spd, -spd, spd, spd)
        else:
            self.motor.set_motor_model(spd, spd, -spd, -spd)

        time.sleep(dur)
        self.wait()

    # -------------------------------------------------------------------------
    # Sensor & Autonomous Modes
    # -------------------------------------------------------------------------

    def get_distance(self) -> Optional[float]:
        """Return the current ultrasonic distance in cm (or None on bad echo)."""
        return self.sonic.get_distance()

    def is_wall_ahead(self, threshold_cm: Optional[float] = None) -> bool:
        """Return True if an obstacle/wall is detected within `threshold_cm`."""
        limit = self.wall_threshold_cm if threshold_cm is None else threshold_cm
        distance = self.get_distance()
        if distance is None or distance <= 0:
            return False
        return distance < limit

    def free_roam(
        self,
        speed: Optional[int] = None,
        wall_threshold_cm: Optional[float] = None,
        poll_interval: float = 0.05,
    ) -> None:
        """
        Free roaming mode:
        1. Drives forward in a straight line.
        2. Continuously checks the ultrasonic sensor for a wall.
        3. When a wall is within `wall_threshold_cm`, stops (wait), turns around
           180 degrees, and continues in a straight line back the other way.
        """
        spd = self.default_speed if speed is None else speed
        limit = self.wall_threshold_cm if wall_threshold_cm is None else wall_threshold_cm

        self.center_servo()
        time.sleep(0.2)

        print(f"Starting free roam (speed={spd}, wall_threshold={limit}cm)...")
        print("Press Ctrl+C to stop.")

        self.run(spd)
        while True:
            distance = self.get_distance()
            if distance is not None and 0 < distance < limit:
                print(f"Wall detected at {distance}cm! Stopping and turning around...")
                self.wait(0.3)
                # Slightly back up if very close to the wall before pivoting
                if distance < 15:
                    self.backward(spd, duration=0.3)
                    self.wait(0.2)
                self.turn_around()
                self.wait(0.3)
                print("Resuming straight line...")
                self.run(spd)

            time.sleep(poll_interval)

    def yolo_roam(
        self,
        speed: Optional[int] = None,
        wall_threshold_cm: Optional[float] = None,
        cls_weights: Optional[str] = None,
        det_weights: Optional[str] = None,
        conf: float = 0.4,
        min_person_height_ratio: float = 0.25,
        go_around_duration: float = 1.6,
    ) -> None:
        """
        YOLO-guided autonomous roaming mode:
        1. Captures live frames from the onboard Raspberry Pi Camera (or USB webcam fallback).
        2. Runs the trained YOLO person detector + standing/walking classifier.
        3. Reacts to pedestrians and obstacles:
           - Wall within `wall_threshold_cm`: stops, turns around 180°, and resumes.
           - WALKING person ahead: stops (`wait`) and yields until they finish crossing/moving.
           - STANDING person ahead (close enough, height_ratio >= min_person_height_ratio):
             stops briefly, then steers around them (`go_around_left` or `go_around_right`
             away from the side they are standing on) and resumes straight driving.
        """
        from detect_pedestrian_action import PedestrianVideoTracker, load_pedestrian_models

        spd = self.default_speed if speed is None else speed
        limit = self.wall_threshold_cm if wall_threshold_cm is None else wall_threshold_cm

        cls_model, det_model, cls_path, det_path = load_pedestrian_models(cls_weights, det_weights)
        print(f"Loading YOLO models:\n  Classifier: {cls_path}\n  Detector:   {det_path}")
        tracker = PedestrianVideoTracker(history_len=10)

        # Initialize camera (Picamera2 on Raspberry Pi, fallback to OpenCV VideoCapture)
        pi_cam = None
        cv_cap = None
        try:
            from camera import Camera
            pi_cam = Camera(stream_size=(640, 480))
            pi_cam.start_stream()
            print("Started Raspberry Pi camera stream (640x480).")
        except Exception as e:
            print(f"Picamera2 unavailable ({e}), falling back to cv2.VideoCapture(0)...")
            cv_cap = cv2.VideoCapture(0)
            if not cv_cap.isOpened():
                raise RuntimeError("Could not open either Picamera2 or cv2.VideoCapture(0).")

        self.center_servo()
        time.sleep(0.2)

        # Warm up both YOLO models BEFORE starting the motors so the robot doesn't
        # drive blindly during first-frame NCNN initialization and doesn't spike
        # battery current (CPU + motor inrush simultaneously).
        print("Warming up YOLO models before starting motors...")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det_model(dummy_frame, classes=[0], conf=conf, verbose=False)
        cls_model(np.zeros((52, 20, 3), dtype=np.uint8), imgsz=64, verbose=False)
        print("YOLO warm-up complete!")

        print(f"Starting YOLO self-driving roam (speed={spd}, wall_threshold={limit}cm)...")
        print("Behavior: WALKING person -> wait | STANDING person -> go around | WALL -> turn around")
        print("Press Ctrl+C to stop.")

        # Ramp motors up gently in 2 steps to prevent battery voltage brownout on Raspberry Pi 5
        half_spd = max(400, spd // 2)
        self.run(half_spd)
        time.sleep(0.15)
        self.run(spd)
        driving = True

        try:
            while True:
                # 1. Check ultrasonic sensor for immediate wall/obstacle
                distance = self.get_distance()
                if distance is not None and 0 < distance < limit:
                    print(f"[ULTRASONIC] Wall/obstacle at {distance:.1f}cm! Turning around...")
                    self.wait(0.3)
                    if distance < 15:
                        self.backward(spd, duration=0.3)
                        self.wait(0.2)
                    self.turn_around()
                    self.wait(0.3)
                    self.run(spd)
                    driving = True
                    continue

                # 2. Grab latest camera frame
                if pi_cam is not None:
                    jpeg_bytes = pi_cam.get_frame()
                    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                else:
                    ret, frame = cv_cap.read()
                    if not ret:
                        time.sleep(0.05)
                        continue

                if frame is None:
                    continue

                # 3. Run YOLO detection + standing/walking classification
                _, detections = tracker.update_and_classify_with_detections(
                    frame, det_model, cls_model, conf=conf, is_video_stream=True
                )

                # Filter for pedestrians roughly in front of the robot (center 70% of FOV)
                # and large/close enough in the frame to require reaction
                relevant = [
                    d for d in detections
                    if 0.15 <= d["center_x_norm"] <= 0.85
                    and d["height_ratio"] >= min_person_height_ratio
                ]

                walking_people = [d for d in relevant if d["action"] == "walking"]
                standing_people = [d for d in relevant if d["action"] == "standing"]

                if standing_people:
                    # First check if anyone is walking -> wait for them
                    if walking_people:
                        closest_walker = max(walking_people, key=lambda d: d["height_ratio"])
                        if driving:
                            print(
                                f"[YOLO] WALKING pedestrian detected (ID={closest_walker['track_id']}, "
                                f"conf={closest_walker['confidence']:.0%}) -> Waiting..."
                            )
                            self.wait()
                            driving = False
                    else:
                        # Someone is standing still in front of the robot -> go around them
                        closest_stander = max(standing_people, key=lambda d: d["height_ratio"])
                        cx = closest_stander["center_x_norm"]
                        self.wait(0.25)
                        if cx >= 0.5:
                            # Person is on the right half of the frame -> go around to the left
                            print(
                                f"[YOLO] STANDING pedestrian ahead-right (ID={closest_stander['track_id']}, "
                                f"conf={closest_stander['confidence']:.0%}) -> Going around LEFT..."
                            )
                            self.go_around_left(outer_speed=1400, inner_speed=500, duration=go_around_duration)
                        else:
                            # Person is on the left half of the frame -> go around to the right
                            print(
                                f"[YOLO] STANDING pedestrian ahead-left (ID={closest_stander['track_id']}, "
                                f"conf={closest_stander['confidence']:.0%}) -> Going around RIGHT..."
                            )
                            self.go_around_right(outer_speed=1400, inner_speed=500, duration=go_around_duration)
                        self.wait(0.25)
                        self.run(half_spd)
                        time.sleep(0.15)
                        self.run(spd)
                        driving = True
                elif walking_people:
                    # Priority 1: Someone is walking in front of the robot -> wait for them
                    closest_walker = max(walking_people, key=lambda d: d["height_ratio"])
                    if driving:
                        print(
                            f"[YOLO] WALKING pedestrian detected (ID={closest_walker['track_id']}, "
                            f"conf={closest_walker['confidence']:.0%}) -> Waiting..."
                        )
                        self.wait()
                        driving = False
                else:
                    # Path is clear of pedestrians -> keep driving forward
                    if not driving:
                        print("[YOLO] Pedestrian cleared -> Resuming forward drive.")
                        self.run(half_spd)
                        time.sleep(0.15)
                        self.run(spd)
                        driving = True
        finally:
            if pi_cam is not None:
                try:
                    pi_cam.stop_stream()
                    pi_cam.close()
                except Exception:
                    pass
            if cv_cap is not None:
                cv_cap.release()

    def close(self) -> None:
        """Stop motors and release hardware resources."""
        try:
            self.wait()
        except Exception:
            pass
        try:
            self.sonic.close()
        except Exception:
            pass
        try:
            self.motor.close()
        except Exception:
            pass


def test_yolo_on_pc(
    source: str = "0",
    cls_weights: Optional[str] = None,
    det_weights: Optional[str] = None,
    conf: float = 0.4,
    min_person_height_ratio: float = 0.25,
    output_path: Optional[str] = None,
    show: bool = True,
    max_frames: Optional[int] = None,
) -> list[dict]:
    """
    Test the YOLO standing/walking pedestrian model and robot self-driving decisions
    directly on a PC without the physical car hardware.

    - Supports a webcam (`source="0"`), a video file (`.mp4`), or an image (`.jpg`/`.png`).
    - Annotates each frame with STANDING / WALKING bounding boxes AND a live HUD
      banner showing what command the self-driving robot would execute:
        * "ROBOT ACTION: WAIT / STOP (Walking pedestrian ahead)"
        * "ROBOT ACTION: GO AROUND LEFT (Standing pedestrian on right)"
        * "ROBOT ACTION: GO AROUND RIGHT (Standing pedestrian on left)"
        * "ROBOT ACTION: DRIVE FORWARD (Path clear)"
    """
    from detect_pedestrian_action import (
        PedestrianVideoTracker,
        VIDEO_EXTENSIONS,
        load_pedestrian_models,
    )

    cls_model, det_model, cls_path, det_path = load_pedestrian_models(cls_weights, det_weights)
    print(f"[PC Test] Loaded YOLO models:\n  Classifier: {cls_path}\n  Detector:   {det_path}")
    tracker = PedestrianVideoTracker(history_len=10)

    def decide_and_draw_hud(frame: np.ndarray, detections: list[dict]) -> tuple[np.ndarray, str]:
        fh, fw = frame.shape[:2]
        relevant = [
            d for d in detections
            if 0.15 <= d["center_x_norm"] <= 0.85 and d["height_ratio"] >= min_person_height_ratio
        ]
        walking = [d for d in relevant if d["action"] == "walking"]
        standing = [d for d in relevant if d["action"] == "standing"]

        if walking:
            w_person = max(walking, key=lambda d: d["height_ratio"])
            robot_cmd = f"WAIT / STOP (Walking person ID={w_person['track_id']} {w_person['confidence']:.0%})"
            hud_color = (0, 0, 220)  # Red banner for stop/yield
        elif standing:
            s_person = max(standing, key=lambda d: d["height_ratio"])
            if s_person["center_x_norm"] >= 0.5:
                robot_cmd = f"GO AROUND LEFT (Standing person ID={s_person['track_id']} on right)"
            else:
                robot_cmd = f"GO AROUND RIGHT (Standing person ID={s_person['track_id']} on left)"
            hud_color = (0, 140, 255)  # Orange banner for steering around
        else:
            robot_cmd = "DRIVE FORWARD (Path clear)"
            hud_color = (0, 160, 0)  # Green banner for driving forward

        # Draw ROI corridor guide lines (15% to 85% width)
        left_x = int(0.15 * fw)
        right_x = int(0.85 * fw)
        cv2.line(frame, (left_x, 45), (left_x, fh), (180, 180, 180), 1, cv2.LINE_AA)
        cv2.line(frame, (right_x, 45), (right_x, fh), (180, 180, 180), 1, cv2.LINE_AA)

        # Draw top HUD bar showing robot action
        cv2.rectangle(frame, (0, 0), (fw, 42), hud_color, -1)
        cv2.putText(
            frame,
            f"ROBOT: {robot_cmd}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame, robot_cmd

    is_webcam = source.isdigit()
    is_video = is_webcam or source.lower().endswith(VIDEO_EXTENSIONS)
    history_log: list[dict] = []

    if is_video:
        cap_src = int(source) if is_webcam else source
        cap = None
        first_frame = None

        if is_webcam:
            # On Windows, try DirectShow (CAP_DSHOW) first to avoid MSMF grabFrame errors (-1072875772),
            # then fall back to default backend, and verify we can actually read a frame.
            backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
            for backend in backends:
                test_cap = cv2.VideoCapture(cap_src, backend)
                if test_cap.isOpened():
                    ret, test_frame = test_cap.read()
                    if ret and test_frame is not None:
                        cap = test_cap
                        first_frame = test_frame
                        break
                    test_cap.release()

            if cap is None:
                # Fallback to sample video in project root if webcam isn't available or can't grab frames
                fallback_video = os.path.join(PROJECT_ROOT, "gettyimages-1477907561-640_adpp.mp4")
                if os.path.exists(fallback_video):
                    print(
                        f"[PC Test] Webcam '{source}' could not grab frames -> "
                        f"falling back to sample video: {fallback_video}"
                    )
                    cap = cv2.VideoCapture(fallback_video)
                    source = fallback_video
                    is_webcam = False
        else:
            cap = cv2.VideoCapture(cap_src)

        if cap is None or not cap.isOpened():
            raise FileNotFoundError(f"Could not open video source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"[PC Test] Running on '{source}' ({width}x{height} @ {fps:.1f} FPS)...")
        if show:
            print("[PC Test] Press 'q' in the video window to quit.")

        frame_idx = 0
        last_printed_cmd = None
        while True:
            if first_frame is not None:
                ret, frame = True, first_frame
                first_frame = None
            else:
                ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            annotated, detections = tracker.update_and_classify_with_detections(
                frame, det_model, cls_model, conf=conf, is_video_stream=True
            )
            annotated, robot_cmd = decide_and_draw_hud(annotated, detections)
            history_log.append({"frame": frame_idx, "robot_cmd": robot_cmd, "detections": detections})

            if robot_cmd != last_printed_cmd:
                print(f"  [Frame {frame_idx:04d}] -> {robot_cmd} (detected={len(detections)})")
                last_printed_cmd = robot_cmd

            if writer is not None:
                writer.write(annotated)

            if show:
                try:
                    cv2.imshow("PC Robot Simulator - YOLO Standing vs Walking", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    print("[PC Test] OpenCV GUI window not supported in this build; continuing in console mode.")
                    show = False

            if max_frames is not None and frame_idx >= max_frames:
                break

        cap.release()
        if writer is not None:
            writer.release()
            print(f"[PC Test] Saved annotated simulation video to: {os.path.abspath(output_path)}")
        if show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
    else:
        img = cv2.imread(source)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {source}")

        annotated, detections = tracker.update_and_classify_with_detections(
            img, det_model, cls_model, conf=conf, is_video_stream=False
        )
        annotated, robot_cmd = decide_and_draw_hud(annotated, detections)
        history_log.append({"frame": 1, "robot_cmd": robot_cmd, "detections": detections})
        print(f"[PC Test] Image '{source}' -> {robot_cmd} | Detections: {detections}")

        if output_path is None:
            output_path = os.path.join(PROJECT_ROOT, "output_pc_test.jpg")
        cv2.imwrite(output_path, annotated)
        print(f"[PC Test] Saved annotated image to: {os.path.abspath(output_path)}")

        if show:
            try:
                cv2.imshow("PC Robot Simulator - YOLO Standing vs Walking", annotated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    return history_log


if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "yolo"

    if mode in ("test_pc", "pc", "sim"):
        # Run PC simulation without initializing car hardware
        src = sys.argv[2] if len(sys.argv) > 2 else "0"
        out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(PROJECT_ROOT, "output_pc_sim.mp4")
        test_yolo_on_pc(source=src, output_path=out, show=True)
        sys.exit(0)

    with RoamCar() as car:
        try:
            if mode == "yolo":
                car.yolo_roam()
            elif mode == "roam":
                car.free_roam()
            elif mode == "left":
                print("Testing go_around_left (arc left for 2s)...")
                car.go_around_left(outer_speed=2000, inner_speed=600, duration=2.0)
            elif mode == "right":
                print("Testing go_around_right (arc right for 2s)...")
                car.go_around_right(outer_speed=2000, inner_speed=600, duration=2.0)
            elif mode == "run":
                print("Testing run (forward 1.5s) -> wait (1s) -> turn_around -> run (1.5s)...")
                car.run(1000, duration=1.5)
                car.wait(1.0)
                car.turn_around()
                car.wait(0.5)
                car.run(1000, duration=1.5)
            elif mode == "turnaround":
                print("Testing 180-degree turn_around...")
                car.turn_around()
            else:
                print(f"Unknown mode '{mode}'. Available modes: yolo, roam, left, right, run, turnaround")
        except KeyboardInterrupt:
            print("\nStopping car...")
