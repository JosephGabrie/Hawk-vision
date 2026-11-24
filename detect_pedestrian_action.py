import argparse
import os
from collections import defaultdict, deque
import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm")
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_model_paths(
    cls_weights: str | None = None,
    det_weights: str | None = None,
    prefer_ncnn: bool = True,
) -> tuple[str, str]:
    """
    Resolves the standing/walking classifier and person detector model paths.
    Prefers exported NCNN directories (`best_ncnn_model` and `yolo11n_ncnn_model`)
    when available and the `ncnn` Python package is installed, falling back to `.pt`.
    """
    has_ncnn_pkg = False
    if prefer_ncnn:
        try:
            import ncnn  # noqa: F401
            has_ncnn_pkg = True
        except ImportError:
            has_ncnn_pkg = False

    if cls_weights is None:
        ncnn_cls = os.path.join(
            PROJECT_DIR, "runs", "classify", "pedestrian_action", "weights", "best_ncnn_model"
        )
        pt_cls = os.path.join(
            PROJECT_DIR, "runs", "classify", "pedestrian_action", "weights", "best.pt"
        )
        cls_weights = ncnn_cls if (has_ncnn_pkg and os.path.isdir(ncnn_cls)) else pt_cls

    if det_weights is None:
        ncnn_det = os.path.join(PROJECT_DIR, "yolo11n_ncnn_model")
        pt_det = os.path.join(PROJECT_DIR, "yolo11n.pt")
        det_weights = ncnn_det if (has_ncnn_pkg and os.path.isdir(ncnn_det)) else pt_det

    return cls_weights, det_weights


def load_pedestrian_models(
    cls_weights: str | None = None,
    det_weights: str | None = None,
    prefer_ncnn: bool = True,
) -> tuple[YOLO, YOLO, str, str]:
    """Loads `(cls_model, det_model, cls_path, det_path)` with explicit task types."""
    cls_path, det_path = resolve_model_paths(cls_weights, det_weights, prefer_ncnn=prefer_ncnn)
    cls_model = YOLO(cls_path, task="classify")
    det_model = YOLO(det_path, task="detect")
    return cls_model, det_model, cls_path, det_path


class PedestrianVideoTracker:
    """
    Combines the trained YOLO standing/walking classifier with multi-frame motion tracking
    so people standing still in a video are not misclassified as walking.
    """

    def __init__(self, history_len: int = 10):
        self.history_len = history_len
        self.positions = defaultdict(lambda: deque(maxlen=history_len))  # track_id -> deque of (cx, cy, w, h)
        self.prev_crops = {}  # track_id -> grayscale normalized crop (32x64)
        self.prob_history = defaultdict(lambda: deque(maxlen=history_len))  # track_id -> deque of P(walking)

    def update_and_classify_with_detections(
        self,
        frame: np.ndarray,
        det_model: YOLO,
        cls_model: YOLO,
        conf: float = 0.4,
        is_video_stream: bool = True,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Runs person detection + standing/walking classification and returns both:
        1. The annotated BGR frame
        2. A list of detection dicts:
           [{"track_id": int, "bbox": (x1, y1, x2, y2), "action": "walking"|"standing",
             "confidence": float, "p_walk": float, "center_x_norm": float, "height_ratio": float}]
        """
        fh, fw = frame.shape[:2]
        detections: list[dict] = []

        if is_video_stream:
            det_results = det_model.track(
                frame, classes=[0], conf=conf, persist=True, tracker="bytetrack.yaml", verbose=False
            )[0]
        else:
            det_results = det_model(frame, classes=[0], conf=conf, verbose=False)[0]

        if det_results.boxes is None or len(det_results.boxes) == 0:
            return frame, detections

        boxes_xyxy = det_results.boxes.xyxy.cpu().numpy()
        track_ids = (
            det_results.boxes.id.int().cpu().tolist()
            if det_results.boxes.id is not None
            else list(range(len(boxes_xyxy)))
        )

        # First pass: compute per-box raw displacements to estimate global camera motion
        raw_displacements = {}
        for box_xyxy, tid in zip(boxes_xyxy, track_ids):
            x1, y1, x2, y2 = box_xyxy
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            pos_hist = self.positions[tid]
            if len(pos_hist) >= 2:
                old_cx, old_cy, _, _ = pos_hist[0]
                raw_displacements[tid] = (cx - old_cx, cy - old_cy)
            pos_hist.append((cx, cy, bw, bh))

        # Subtract median scene shift (handles slight camera shake/pan)
        if len(raw_displacements) >= 2:
            med_dx = float(np.median([d[0] for d in raw_displacements.values()]))
            med_dy = float(np.median([d[1] for d in raw_displacements.values()]))
        else:
            med_dx, med_dy = 0.0, 0.0

        for box_xyxy, tid in zip(boxes_xyxy, track_ids):
            x1, y1, x2, y2 = map(int, box_xyxy.tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # 1. Match the native resolution (~20x52) of the Pedestrians training dataset
            crop_matched = cv2.resize(crop, (20, 52), interpolation=cv2.INTER_AREA)
            cls_res = cls_model(crop_matched, imgsz=64, verbose=False)[0]
            probs = cls_res.probs.data.cpu().numpy()
            # Map by class name
            names = cls_res.names
            standing_idx = 0 if names[0] == "standing" else 1
            walking_idx = 1 - standing_idx
            p_walk_cls = float(probs[walking_idx])

            # Check if lower legs/feet are cut off at the bottom of the frame (close-up / portrait shot)
            feet_cutoff = (y2 >= fh - 6) and (bh > 0.45 * fh)
            if feet_cutoff and not is_video_stream:
                # In a single portrait/close-up frame where feet are cut off, diminish walking prior
                p_walk_cls *= 0.25

            if is_video_stream:
                # 2. Compute in-box limb/body motion (focus on inner 60% width and lower 75% height)
                gray_norm = cv2.cvtColor(cv2.resize(crop, (32, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                roi = gray_norm[16:60, 6:26].astype(np.float32)

                limb_motion = 0.0
                if tid in self.prev_crops:
                    prev_roi = self.prev_crops[tid]
                    diff = np.mean(np.abs(roi - prev_roi))
                    # Subtract baseline sensor noise (~3.5 intensity levels)
                    limb_motion = max(0.0, (diff - 3.5) / 14.0)
                    limb_motion = min(1.0, limb_motion)
                self.prev_crops[tid] = roi

                # 3. Compute trajectory translation speed relative to person height
                pos_hist = self.positions[tid]
                if tid in raw_displacements and len(pos_hist) >= 2:
                    dx = raw_displacements[tid][0] - med_dx
                    dy = raw_displacements[tid][1] - med_dy
                    dist = float(np.hypot(dx, dy))
                    frames_elapsed = max(1, len(pos_hist) - 1)
                    # Normalized speed per frame (% of body height moved per frame)
                    rel_speed = (dist / frames_elapsed) / float(bh)
                    # Walking typically moves >= 0.6% of body height per frame
                    traj_motion = min(1.0, max(0.0, (rel_speed - 0.0025) / 0.008))
                else:
                    traj_motion = 0.0

                motion_score = max(limb_motion, traj_motion)

                # If we have at least 2 frames of history for this person:
                # When someone is standing still (motion_score ~ 0), gate the walking probability
                # so static people are classified as STANDING.
                if len(pos_hist) >= 2:
                    p_walk = motion_score * (0.45 + 0.55 * p_walk_cls)
                else:
                    p_walk = 0.35 * p_walk_cls if feet_cutoff else p_walk_cls
            else:
                p_walk = p_walk_cls

            self.prob_history[tid].append(p_walk)
            avg_p_walk = float(np.mean(self.prob_history[tid]))

            if avg_p_walk >= 0.5:
                action = "walking"
                action_conf = avg_p_walk
                color = (0, 200, 0)  # Green for walking
            else:
                action = "standing"
                action_conf = 1.0 - avg_p_walk
                color = (0, 140, 255)  # Orange for standing

            label_text = f"{action.upper()} {action_conf:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = max(th + 8, y1)
            cv2.rectangle(frame, (x1, ty - th - 8), (x1 + tw + 6, ty), color, -1)
            cv2.putText(
                frame,
                label_text,
                (x1 + 3, ty - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            detections.append(
                {
                    "track_id": tid,
                    "bbox": (x1, y1, x2, y2),
                    "action": action,
                    "confidence": action_conf,
                    "p_walk": avg_p_walk,
                    "center_x_norm": 0.5 * (x1 + x2) / max(1, fw),
                    "height_ratio": bh / max(1, fh),
                }
            )

        return frame, detections

    def update_and_classify(
        self,
        frame: np.ndarray,
        det_model: YOLO,
        cls_model: YOLO,
        conf: float = 0.4,
        is_video_stream: bool = True,
    ) -> np.ndarray:
        annotated, _ = self.update_and_classify_with_detections(
            frame, det_model, cls_model, conf=conf, is_video_stream=is_video_stream
        )
        return annotated


def run_inference(
    source: str,
    cls_weights: str | None = None,
    det_weights: str | None = None,
    output_path: str | None = None,
    conf: float = 0.4,
    show: bool = True,
):
    cls_model, det_model, cls_path, det_path = load_pedestrian_models(cls_weights, det_weights)
    print(f"Loaded models:\n  Classifier: {cls_path}\n  Detector:   {det_path}")
    tracker = PedestrianVideoTracker(history_len=10)

    is_webcam = source.isdigit()
    is_video = is_webcam or source.lower().endswith(VIDEO_EXTENSIONS)

    if is_video:
        cap_source = int(source) if is_webcam else source
        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if output_path is None:
            output_path = "output_video.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Processing video '{source}' ({width}x{height} @ {fps:.1f} FPS)...")
        print("Press 'q' in the video window to stop early.")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            annotated = tracker.update_and_classify(
                frame, det_model, cls_model, conf=conf, is_video_stream=True
            )
            writer.write(annotated)

            if show:
                cv2.imshow("Pedestrian Action Detection (Standing vs Walking)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_idx % 30 == 0:
                if total_frames > 0:
                    print(f"Processed {frame_idx}/{total_frames} frames...")
                else:
                    print(f"Processed {frame_idx} frames...")

        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()
        print(f"Done! Annotated video saved to: {os.path.abspath(output_path)}")

    else:
        img = cv2.imread(source)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {source}")

        h, w = img.shape[:2]
        if w < 160 and h < 320:
            crop_matched = cv2.resize(img, (20, 52), interpolation=cv2.INTER_AREA)
            res = cls_model(crop_matched, imgsz=64, verbose=False)[0]
            top1_idx = int(res.probs.top1)
            label = res.names[top1_idx]
            score = float(res.probs.top1conf)
            print(f"{source} -> {label} ({score:.2%})")
            return label, score

        if output_path is None:
            output_path = "output_prediction.jpg"

        annotated = tracker.update_and_classify(
            img, det_model, cls_model, conf=conf, is_video_stream=False
        )
        cv2.imwrite(output_path, annotated)
        print(f"Saved annotated image to: {os.path.abspath(output_path)}")
        if show:
            cv2.imshow("Pedestrian Action Detection", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run pedestrian standing vs walking detection on a video, webcam, or image."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to video file (e.g. video.mp4), '0' for webcam, or path to an image",
    )
    parser.add_argument(
        "--cls-weights",
        default=None,
        help="Path to your trained standing/walking YOLO weights (defaults to NCNN model if available, else best.pt)",
    )
    parser.add_argument(
        "--det-weights",
        default=None,
        help="YOLO detector weights for finding people in each video frame (defaults to NCNN model if available, else yolo11n.pt)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (defaults to output_video.mp4 for video, output_prediction.jpg for image)",
    )
    parser.add_argument("--conf", type=float, default=0.4, help="Person detection confidence threshold")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable live popup window (useful when only saving the output video)",
    )
    args = parser.parse_args()
    run_inference(
        source=args.source,
        cls_weights=args.cls_weights,
        det_weights=args.det_weights,
        output_path=args.output,
        conf=args.conf,
        show=not args.no_show,
    )
