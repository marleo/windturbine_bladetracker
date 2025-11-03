"""
Wind Turbine Blade Tracking Application

This application uses YOLO segmentation and ByteTrack to track wind turbine blades
with angle-based ID stabilization for consistent blade identification through occlusions.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import math
import argparse
import time
import threading
import queue
import itertools
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class Detection:
    """Represents a detected blade with its properties."""
    id: int
    centroid: Tuple[float, float]
    polygon: Optional[np.ndarray] = None


@dataclass
class PerformanceMetrics:
    """Container for performance timing data."""
    inference_time: float
    processing_time: float
    hub_estimation_time: float
    id_stabilization_time: float
    rendering_time: float
    fps: float


class AngleIDStabilizer:
    """
    Stabilizes blade IDs using angle-based prediction to maintain consistent
    blade identification through occlusions.
    """

    def __init__(self, n_blades: int = 3, max_missing_frames: int = 10, smoothing_alpha: float = 0.7):
        self.n_blades = n_blades
        self.max_missing_frames = max_missing_frames
        self.smoothing_alpha = smoothing_alpha

        self.stable_id_counter = 1
        self.last_known_angles: Dict[int, float] = {}
        self.angular_velocities: Dict[int, float] = {}
        self.missing_frames: Dict[int, int] = {}

    def _angle_difference(self, angle1: float, angle2: float) -> float:
        """Calculate the smallest angle difference between two angles."""
        diff = (angle1 - angle2 + math.pi) % (2 * math.pi) - math.pi
        return diff

    def _predict_angle(self, stable_id: int) -> float:
        """Predict the current angle for a stable ID based on last known position and velocity."""
        if stable_id not in self.last_known_angles:
            return 0.0

        last_angle = self.last_known_angles[stable_id]
        velocity = self.angular_velocities.get(stable_id, 0.0)

        predicted_angle = last_angle + velocity
        # Normalize to [-pi, pi]
        predicted_angle = (predicted_angle + math.pi) % (2 * math.pi) - math.pi

        return predicted_angle

    def stabilize_ids(self, detections: List[Detection], hub_center: Tuple[float, float]) -> Dict[int, int]:
        """
        Assign stable IDs to detections based on angular position relative to hub.

        Args:
            detections: List of detected blades
            hub_center: (x, y) coordinates of the hub center

        Returns:
            Dictionary mapping tracker IDs to stable blade IDs
        """
        hub_x, hub_y = hub_center

        # Calculate current angles for all detections
        current_angles = {}
        tracker_ids = set()

        for detection in detections:
            tracker_ids.add(detection.id)
            center_x, center_y = detection.centroid
            angle = math.atan2(center_y - hub_y, center_x - hub_x)
            current_angles[detection.id] = angle

        # Update missing frame counters and predict angles
        predicted_angles = {}
        for stable_id in list(self.last_known_angles.keys()):
            if self.missing_frames.get(stable_id, 0) < self.max_missing_frames:
                predicted_angles[stable_id] = self._predict_angle(stable_id)
                self.missing_frames[stable_id] += 1

        # Match current detections to predicted positions
        assignments = {}
        unassigned_trackers = list(tracker_ids)
        assigned_stable_ids = set()

        while unassigned_trackers and len(assigned_stable_ids) < self.n_blades:
            best_match = None
            min_angle_diff = float('inf')

            for stable_id, predicted_angle in predicted_angles.items():
                if stable_id in assigned_stable_ids:
                    continue

                for tracker_id in unassigned_trackers:
                    current_angle = current_angles[tracker_id]
                    angle_diff = abs(self._angle_difference(current_angle, predicted_angle))

                    if angle_diff < min_angle_diff:
                        min_angle_diff = angle_diff
                        best_match = (tracker_id, stable_id)

            if best_match:
                tracker_id, stable_id = best_match

                # Update velocity estimate
                last_angle = self.last_known_angles.get(stable_id, current_angles[tracker_id])
                current_angle = current_angles[tracker_id]
                angle_diff = self._angle_difference(current_angle, last_angle)

                old_velocity = self.angular_velocities.get(stable_id, 0.0)
                new_velocity = (self.smoothing_alpha * angle_diff +
                              (1 - self.smoothing_alpha) * old_velocity)

                self.angular_velocities[stable_id] = new_velocity
                self.last_known_angles[stable_id] = current_angle
                self.missing_frames[stable_id] = 0

                assignments[tracker_id] = stable_id
                unassigned_trackers.remove(tracker_id)
                assigned_stable_ids.add(stable_id)

                if stable_id in predicted_angles:
                    del predicted_angles[stable_id]
            else:
                break

        # Assign new IDs to remaining unassigned trackers
        for tracker_id in unassigned_trackers:
            if self.stable_id_counter <= self.n_blades:
                assignments[tracker_id] = self.stable_id_counter
                self.last_known_angles[self.stable_id_counter] = current_angles[tracker_id]
                self.angular_velocities[self.stable_id_counter] = 0.0
                self.missing_frames[self.stable_id_counter] = 0
                self.stable_id_counter += 1
            else:
                assignments[tracker_id] = -1  # Invalid ID

        return assignments


class GeometryUtils:
    """Utility functions for geometric calculations."""

    @staticmethod
    def calculate_mask_centroid(mask_polygon: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Calculate the centroid of a polygon using the shoelace formula.

        Args:
            mask_polygon: Nx2 array of polygon vertices

        Returns:
            (x, y) centroid coordinates or None if calculation fails
        """
        if mask_polygon is None or len(mask_polygon) == 0:
            return None

        x_coords = mask_polygon[:, 0]
        y_coords = mask_polygon[:, 1]

        # Shoelace formula
        area = np.dot(x_coords, np.roll(y_coords, -1)) - np.dot(y_coords, np.roll(x_coords, -1))

        if abs(area) < 1e-6:
            # Degenerate polygon, return mean
            return float(np.mean(x_coords)), float(np.mean(y_coords))

        # Calculate centroid
        cx = np.sum((x_coords + np.roll(x_coords, -1)) *
                   (x_coords * np.roll(y_coords, -1) - y_coords * np.roll(x_coords, -1))) / (3 * area)
        cy = np.sum((y_coords + np.roll(y_coords, -1)) *
                   (x_coords * np.roll(y_coords, -1) - y_coords * np.roll(x_coords, -1))) / (3 * area)

        return float(cx), float(cy)

    @staticmethod
    def fit_line_to_mask(mask_polygon: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Fit a line to a mask polygon using OpenCV's fitLine.

        Args:
            mask_polygon: Nx2 array of polygon vertices

        Returns:
            Tuple of (direction_vector, point_on_line) or None
        """
        if mask_polygon is None or len(mask_polygon) < 2:
            return None

        points = mask_polygon.astype(np.float32)
        [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

        direction = np.array([vx[0], vy[0]])
        point = np.array([x0[0], y0[0]])

        return direction, point

    @staticmethod
    def find_line_intersection(line1_dir: np.ndarray, line1_point: np.ndarray,
                             line2_dir: np.ndarray, line2_point: np.ndarray) -> Optional[np.ndarray]:
        """
        Find intersection point of two lines.

        Args:
            line1_dir: Direction vector of first line
            line1_point: Point on first line
            line2_dir: Direction vector of second line
            line2_point: Point on second line

        Returns:
            Intersection point or None if lines are parallel
        """
        # Solve: line1_point + t * line1_dir = line2_point + s * line2_dir
        A = np.array([line1_dir, -line2_dir]).T
        b = line2_point - line1_point

        try:
            ts = np.linalg.solve(A, b)
            intersection = line1_point + ts[0] * line1_dir
            return intersection
        except np.linalg.LinAlgError:
            return None


class HubEstimator:
    """Estimates the hub position from blade mask intersections."""

    def __init__(self, smoothing_alpha: float = 0.1):
        self.smoothing_alpha = smoothing_alpha
        self.smoothed_position: Optional[np.ndarray] = None

    def estimate_hub_position(self, detections: List[Detection], frame_width: int, frame_height: int) -> Tuple[float, float]:
        """
        Estimate hub position by finding intersections of blade mask lines.

        Args:
            detections: List of blade detections
            frame_width: Width of the video frame
            frame_height: Height of the video frame

        Returns:
            (x, y) coordinates of estimated hub position
        """
        # Extract lines from blade masks
        mask_lines = []
        for detection in detections:
            if detection.polygon is not None:
                polygon = detection.polygon.reshape(-1, 2).astype(np.float32)
                line_params = GeometryUtils.fit_line_to_mask(polygon)
                if line_params is not None:
                    mask_lines.append(line_params)

        # Find intersections between all pairs of lines
        intersections = []
        if len(mask_lines) >= 2:
            for i, j in itertools.combinations(range(len(mask_lines)), 2):
                line1_dir, line1_pt = mask_lines[i]
                line2_dir, line2_pt = mask_lines[j]

                intersection = GeometryUtils.find_line_intersection(line1_dir, line1_pt, line2_dir, line2_pt)
                if (intersection is not None and
                    0 <= intersection[0] < frame_width and
                    0 <= intersection[1] < frame_height):
                    intersections.append(intersection)

        # Update smoothed hub position
        if intersections:
            new_position = np.mean(intersections, axis=0)

            if self.smoothed_position is None:
                self.smoothed_position = new_position
            else:
                self.smoothed_position = (self.smoothing_alpha * new_position +
                                        (1 - self.smoothing_alpha) * self.smoothed_position)

        # Return current estimate (fallback to center if no estimate available)
        if self.smoothed_position is not None:
            return tuple(self.smoothed_position)
        else:
            return (frame_width / 2, frame_height / 2)


class PerformanceMonitor:
    """Monitors and reports performance metrics."""

    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []

    def record_metrics(self, inference_time: float, processing_time: float,
                      hub_estimation_time: float, id_stabilization_time: float,
                      rendering_time: float, fps: float):
        """Record a set of performance metrics."""
        metrics = PerformanceMetrics(
            inference_time=inference_time,
            processing_time=processing_time,
            hub_estimation_time=hub_estimation_time,
            id_stabilization_time=id_stabilization_time,
            rendering_time=rendering_time,
            fps=fps
        )
        self.metrics_history.append(metrics)

    def print_summary(self):
        """Print performance summary statistics."""
        if not self.metrics_history:
            return

        print(f"\n\nPerformance Summary ({len(self.metrics_history)} frames processed):")
        print("=" * 60)

        # Calculate averages
        avg_inference = np.mean([m.inference_time for m in self.metrics_history]) * 1000
        avg_processing = np.mean([m.processing_time for m in self.metrics_history]) * 1000
        avg_hub_est = np.mean([m.hub_estimation_time for m in self.metrics_history]) * 1000
        avg_id_stab = np.mean([m.id_stabilization_time for m in self.metrics_history]) * 1000
        avg_rendering = np.mean([m.rendering_time for m in self.metrics_history]) * 1000
        avg_fps = np.mean([m.fps for m in self.metrics_history])

        print(f"Average Inference Time:     {avg_inference:.2f} ms")
        print(f"Average Processing Time:    {avg_processing:.2f} ms")
        print(f"Average Hub Estimation:     {avg_hub_est:.2f} ms")
        print(f"Average ID Stabilization:   {avg_id_stab:.2f} ms")
        print(f"Average Rendering Time:     {avg_rendering:.2f} ms")
        print(f"Average FPS:               {avg_fps:.1f}")

        total_avg_time = avg_inference + avg_processing + avg_hub_est + avg_id_stab + avg_rendering
        print(f"Total Average Time:         {total_avg_time:.2f} ms")


class VideoProcessor:
    """Handles video input/output operations."""

    def __init__(self, video_path: Optional[str] = None, output_path: Optional[str] = None):
        self.video_path = video_path
        self.output_path = output_path
        self.capture: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0.0
        self.total_frames = 0

    def initialize(self) -> bool:
        """Initialize video capture and writer. Returns True if successful."""
        if self.video_path:
            # Video file input
            self.capture = cv2.VideoCapture(self.video_path)
            if not self.capture.isOpened():
                print(f"Error: Could not open video file: {self.video_path}")
                return False

            self.fps = self.capture.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if self.output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps,
                                            (self.frame_width, self.frame_height))

            print(f"Processing video: {self.video_path}")
            if self.output_path:
                print(f"Output: {self.output_path}")
            print(f"Resolution: {self.frame_width}x{self.frame_height}, FPS: {self.fps}, Frames: {self.total_frames}")

        else:
            # Webcam input
            self.capture = cv2.VideoCapture(2)  # Use camera index 2
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not self.capture.isOpened():
                print("Error: Could not open webcam")
                return False

            self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = 30.0  # Assume 30 FPS for webcam

        return True

    def read_frame(self) -> Optional[np.ndarray]:
        """Read next frame from video. Returns None if no more frames."""
        if self.capture is None:
            return None

        ret, frame = self.capture.read()
        return frame if ret else None

    def write_frame(self, frame: np.ndarray):
        """Write frame to output video."""
        if self.writer is not None:
            self.writer.write(frame)

    def release(self):
        """Release video resources."""
        if self.capture is not None:
            self.capture.release()
        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()


class BladeTracker:
    """Main blade tracking application using YOLO and ByteTrack."""

    # Configuration constants
    MODEL_SIZE = 640
    CONFIDENCE_THRESHOLD = 0.15
    IOU_THRESHOLD = 0.5
    TRACKER_CONFIG = "bytetrack.yaml"
    NUM_BLADES = 3
    HUB_SMOOTHING_ALPHA = 0.1

    # Visualization constants
    BLADE_COLORS = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (0, 255, 255),  # Yellow
        (255, 0, 255)   # Magenta
    ]
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.8
    FONT_THICKNESS = 1

    def __init__(self, model_path: str, headless: bool = False, debug: bool = False, no_render: bool = False):
        self.model_path = model_path
        self.headless = headless
        self.debug = debug
        self.no_render = no_render

        # Initialize components
        self.model: Optional[YOLO] = None
        self.id_stabilizer = AngleIDStabilizer(n_blades=self.NUM_BLADES)
        self.hub_estimator = HubEstimator(smoothing_alpha=self.HUB_SMOOTHING_ALPHA)
        self.performance_monitor = PerformanceMonitor()

        # State variables
        self.frame_count = 0
        self.actual_device = "unknown"

    def initialize(self) -> bool:
        """Initialize the YOLO model and print device information."""
        try:
            print(f"Loading model: {self.model_path}")
            self.model = YOLO(self.model_path, task='segment')

            # Print device information
            device_info = str(self.model.device)
            print(f"Model device: {device_info}")

            # Check for GPU usage
            has_cuda_params = any(param.is_cuda for param in self.model.parameters())

            if has_cuda_params or 'cuda' in device_info.lower():
                print(f"Inference device: GPU ({device_info})")
            else:
                print(f"Inference device: CPU ({device_info})")
                self._check_gpu_availability()

            return True

        except Exception as e:
            print(f"Error initializing model: {e}")
            return False

    def _check_gpu_availability(self):
        """Check and report GPU availability."""
        print("Checking for GPU availability...")
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            print(f"PyTorch CUDA available: {cuda_available}")
            if cuda_available:
                device_count = torch.cuda.device_count()
                print(f"CUDA devices: {device_count}")
                for i in range(device_count):
                    print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
                print("Note: Model may still use GPU for inference even if device shows CPU")
            else:
                print("No CUDA support detected")
        except ImportError:
            print("PyTorch not available")

    def _detect_device_from_results(self, results) -> str:
        """Determine the actual inference device from YOLO results."""
        try:
            r0 = results[0]

            # Try multiple ways to get device info
            if hasattr(r0, 'boxes') and r0.boxes is not None:
                if hasattr(r0.boxes, 'device'):
                    device = str(r0.boxes.device)
                elif hasattr(r0.boxes, 'xyxy') and hasattr(r0.boxes.xyxy, 'device'):
                    device = str(r0.boxes.xyxy.device)
                else:
                    device = None

            if device is None and hasattr(r0, 'masks') and r0.masks is not None:
                if hasattr(r0.masks, 'device'):
                    device = str(r0.masks.device)
                elif hasattr(r0.masks, 'xy') and len(r0.masks.xy) > 0:
                    try:
                        import torch
                        if isinstance(r0.masks.xy[0], torch.Tensor):
                            device = str(r0.masks.xy[0].device)
                        else:
                            device = None
                    except:
                        device = None
                else:
                    device = None

            if device:
                return "GPU" if 'cuda' in device.lower() else "CPU"
            else:
                return "unknown"

        except Exception:
            return "unknown"

    def _process_detections(self, results) -> List[Detection]:
        """Process YOLO results into Detection objects."""
        detections = []
        r0 = results[0]

        if r0.boxes and r0.boxes.id is not None:
            track_ids = r0.boxes.id.int().cpu().tolist()

            if r0.masks and len(r0.masks.xy) > 0:
                # Use mask centroids
                for i, mask_poly in enumerate(r0.masks.xy):
                    centroid = GeometryUtils.calculate_mask_centroid(np.array(mask_poly, dtype=np.float32))
                    if centroid:
                        polygon = np.array(mask_poly, dtype=np.int32).reshape((-1, 1, 2))
                        detections.append(Detection(
                            id=track_ids[i],
                            centroid=centroid,
                            polygon=polygon
                        ))
            else:
                # Fallback to bounding box centroids
                for i, bbox in enumerate(r0.boxes.xyxy.cpu().numpy()):
                    x1, y1, x2, y2 = bbox
                    centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                    detections.append(Detection(
                        id=track_ids[i],
                        centroid=centroid,
                        polygon=None
                    ))

        return detections

    def _render_frame(self, frame: np.ndarray, detections: List[Detection],
                     stable_ids: Dict[int, int], hub_center: Tuple[float, float],
                     metrics: PerformanceMetrics) -> np.ndarray:
        """Render detections, hub, and performance metrics on the frame."""
        annotated_frame = frame.copy()

        # Draw detections
        for detection in detections:
            stable_id = stable_ids.get(detection.id)
            if stable_id and stable_id > 0:
                color = self.BLADE_COLORS[(stable_id - 1) % len(self.BLADE_COLORS)]
                center_x, center_y = detection.centroid

                # Draw mask overlay if available
                if detection.polygon is not None:
                    overlay = annotated_frame.copy()
                    cv2.fillPoly(overlay, [detection.polygon], color)
                    cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0, annotated_frame)

                # Draw label and center point
                label = f"Blade {stable_id}"
                cv2.putText(annotated_frame, label, (int(center_x) + 6, int(center_y) - 6),
                           self.FONT, self.FONT_SCALE, (0, 0, 0), self.FONT_THICKNESS + 2, cv2.LINE_AA)
                cv2.putText(annotated_frame, label, (int(center_x) + 6, int(center_y) - 6),
                           self.FONT, self.FONT_SCALE, color, self.FONT_THICKNESS, cv2.LINE_AA)
                cv2.circle(annotated_frame, (int(center_x), int(center_y)), 4, color, -1)

        # Draw hub
        hub_x, hub_y = hub_center
        cv2.circle(annotated_frame, (int(hub_x), int(hub_y)), 5, (255, 255, 255), -1)
        cv2.circle(annotated_frame, (int(hub_x), int(hub_y)), 9, (0, 0, 0), 2)

        # Draw performance metrics if debug mode
        if self.debug:
            y_offset = 30
            cv2.putText(annotated_frame, f"FPS: {metrics.fps:.1f}", (10, y_offset),
                        self.FONT, self.FONT_SCALE, (0, 255, 0), self.FONT_THICKNESS, cv2.LINE_AA)
            y_offset += 30
            cv2.putText(annotated_frame, f"Inference: {metrics.inference_time*1000:.1f}ms", (10, y_offset),
                        self.FONT, self.FONT_SCALE, (255, 255, 0), self.FONT_THICKNESS, cv2.LINE_AA)
            y_offset += 30
            cv2.putText(annotated_frame, f"Processing: {metrics.processing_time*1000:.1f}ms", (10, y_offset),
                        self.FONT, self.FONT_SCALE, (255, 255, 0), self.FONT_THICKNESS, cv2.LINE_AA)
            y_offset += 30
            cv2.putText(annotated_frame, f"Hub Est: {metrics.hub_estimation_time*1000:.1f}ms", (10, y_offset),
                        self.FONT, self.FONT_SCALE, (255, 255, 0), self.FONT_THICKNESS, cv2.LINE_AA)
            y_offset += 30
            cv2.putText(annotated_frame, f"ID Stab: {metrics.id_stabilization_time*1000:.1f}ms", (10, y_offset),
                        self.FONT, self.FONT_SCALE, (255, 255, 0), self.FONT_THICKNESS, cv2.LINE_AA)
            y_offset += 30
            cv2.putText(annotated_frame, f"Rendering: {metrics.rendering_time*1000:.1f}ms", (10, y_offset),
                        self.FONT, self.FONT_SCALE, (255, 255, 0), self.FONT_THICKNESS, cv2.LINE_AA)

        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, PerformanceMetrics, Dict[int, int]]:
        """Process a single frame and return annotated frame, metrics, and stable IDs."""
        height, width = frame.shape[:2]

        # Measure inference time
        inference_start = time.time()
        results = self.model.track(
            frame,
            imgsz=self.MODEL_SIZE,
            conf=self.CONFIDENCE_THRESHOLD,
            iou=self.IOU_THRESHOLD,
            tracker=self.TRACKER_CONFIG,
            persist=True,
            verbose=False
        )
        inference_time = time.time() - inference_start
        fps = 1.0 / inference_time if inference_time > 0 else 0

        # Detect actual device on first frame
        if self.frame_count == 0:
            self.actual_device = self._detect_device_from_results(results)
            print(f"✓ Actual inference device: {self.actual_device}")

        # Process detections
        processing_start = time.time()
        detections = self._process_detections(results)
        processing_time = time.time() - processing_start

        # Estimate hub position
        hub_estimation_start = time.time()
        hub_center = self.hub_estimator.estimate_hub_position(detections, width, height)
        hub_estimation_time = time.time() - hub_estimation_start

        # Stabilize IDs
        id_stabilization_start = time.time()
        stable_ids = self.id_stabilizer.stabilize_ids(detections, hub_center)
        id_stabilization_time = time.time() - id_stabilization_start

        # Render frame
        rendering_start = time.time()
        if not self.no_render:
            annotated_frame = self._render_frame(frame, detections, stable_ids, hub_center,
                                               PerformanceMetrics(inference_time, processing_time,
                                                                hub_estimation_time, id_stabilization_time,
                                                                0.0, fps))
            rendering_time = time.time() - rendering_start
        else:
            annotated_frame = frame
            rendering_time = 0.0

        # Update metrics
        metrics = PerformanceMetrics(
            inference_time=inference_time,
            processing_time=processing_time,
            hub_estimation_time=hub_estimation_time,
            id_stabilization_time=id_stabilization_time,
            rendering_time=rendering_time,
            fps=fps
        )

        if self.no_render:
            self.performance_monitor.record_metrics(
                inference_time, processing_time, hub_estimation_time,
                id_stabilization_time, rendering_time, fps
            )

        return annotated_frame, metrics, stable_ids

    def run(self, video_processor: VideoProcessor):
        """Main processing loop."""
        print("Starting blade tracking (non-blocking pipeline)...")

        # Bounded queues to keep latency low and avoid unbounded memory use.
        frame_queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=3)
        result_queue: "queue.Queue[Optional[Tuple[np.ndarray, PerformanceMetrics, Dict[int,int]]]]" = queue.Queue(maxsize=3)
        stop_event = threading.Event()

        def inference_worker():
            """Worker that consumes frames, runs inference & postprocessing, and pushes results to result_queue."""
            while not stop_event.is_set():
                try:
                    item = frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if item is None:
                    # Sentinel received -> shutdown
                    break

                frame, seq = item

                # Run full processing for the frame (inference + postprocessing)
                try:
                    annotated_frame, metrics, stable_ids = self.process_frame(frame)
                    # Put results for rendering/writing
                    try:
                        result_queue.put_nowait((annotated_frame, metrics, stable_ids, seq))
                    except queue.Full:
                        # Drop oldest result to make room (keep latest)
                        try:
                            _ = result_queue.get_nowait()
                            result_queue.put_nowait((annotated_frame, metrics, stable_ids, seq))
                        except queue.Empty:
                            pass
                except Exception as e:
                    print(f"Inference worker error: {e}")

            # Ensure renderer can stop
            try:
                result_queue.put_nowait(None)
            except Exception:
                pass

        def renderer_worker():
            """Worker that consumes processed frames and handles display / file writing / metrics printing."""
            last_seq = -1
            while not stop_event.is_set():
                try:
                    item = result_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if item is None:
                    break

                annotated_frame, metrics, stable_ids, seq = item

                # If frames arrive out of order drop older ones (keep last)
                if seq <= last_seq:
                    continue
                last_seq = seq

                # Handle output (display or write)
                if video_processor.video_path and not self.no_render:
                    video_processor.write_frame(annotated_frame)
                    print(f"\rProcessing frame {seq}/{video_processor.total_frames} (FPS: {metrics.fps:.1f})", end="", flush=True)

                elif not self.headless and not self.no_render:
                    cv2.imshow('Wind Turbine Blade Tracking', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_event.set()
                        break

                elif self.no_render:
                    if video_processor.video_path:
                        if seq % 10 == 0 or seq == video_processor.total_frames:
                            print(f"\rProcessing frame {seq}/{video_processor.total_frames}", end="", flush=True)
                    else:
                        tracked_blades = sorted(list(set(id for id in stable_ids.values() if id > 0)))
                        tracked_blades_str = ", ".join(map(str, tracked_blades))
                        print(f"\rFPS: {metrics.fps:<5.1f} | "
                              f"Blades: [{tracked_blades_str:<7}] | "
                              f"Infer: {metrics.inference_time*1000:<5.1f}ms | "
                              f"Proc: {metrics.processing_time*1000:<5.1f}ms | "
                              f"Hub: {metrics.hub_estimation_time*1000:<5.1f}ms | "
                              f"ID: {metrics.id_stabilization_time*1000:<5.1f}ms   ",
                              end="", flush=True)

                else:
                    time.sleep(0.001)

            # Renderer exiting
            return

        # Start workers
        infer_thread = threading.Thread(target=inference_worker, name="inference_worker", daemon=True)
        render_thread = threading.Thread(target=renderer_worker, name="renderer_worker", daemon=True)
        infer_thread.start()
        render_thread.start()

        seq = 0
        try:
            while True:
                frame = video_processor.read_frame()
                if frame is None:
                    # Signal shutdown to inference
                    try:
                        frame_queue.put_nowait(None)
                    except Exception:
                        pass
                    break

                self.frame_count += 1
                seq += 1

                # Try to push frame into queue without blocking. If full, drop the oldest frame to keep latency low.
                try:
                    frame_queue.put_nowait((frame, seq))
                except queue.Full:
                    try:
                        _ = frame_queue.get_nowait()  # drop oldest
                        frame_queue.put_nowait((frame, seq))
                    except Exception:
                        pass

                # Small sleep to yield CPU (capture loop stays responsive)
                time.sleep(0.001)

                # Check for user-requested stop (window close)
                if stop_event.is_set():
                    break

        finally:
            # Request shutdown
            stop_event.set()
            try:
                frame_queue.put_nowait(None)
            except Exception:
                pass
            try:
                result_queue.put_nowait(None)
            except Exception:
                pass

            infer_thread.join(timeout=2.0)
            render_thread.join(timeout=2.0)

        # Print performance summary for no-render mode
        if self.no_render:
            self.performance_monitor.print_summary()

        if video_processor.video_path and not self.no_render:
            print(f"\nVideo processing completed. Output saved to: {video_processor.output_path}")


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description='Wind Turbine Blade Tracking with YOLO and ByteTrack',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --video input.mp4 --output output.mp4 --debug
  python main.py --norender --video input.mp4  # Performance benchmarking
  python main.py  # Webcam mode
        """
    )
    parser.add_argument('--headless', action='store_true',
                       help='Run without UI (webcam mode only)')
    parser.add_argument('--video', type=str,
                       help='Path to input video file (optional)')
    parser.add_argument('--output', type=str,
                       help='Path to output video file (required when using --video)')
    parser.add_argument('--debug', action='store_true',
                       help='Show performance metrics overlay')
    parser.add_argument('--model', type=str, default='yolov11s.pt',
                       help='Path to YOLO model file (default: yolov11s.pt)')
    parser.add_argument('--norender', action='store_true',
                       help='Skip video rendering and output performance metrics to console')

    args = parser.parse_args()

    # Validate arguments
    if args.video and not args.output and not args.norender:
        parser.error("--output is required when using --video (unless --norender is specified)")
    if args.headless and args.video:
        parser.error("--headless cannot be used with --video")

    # Initialize components
    video_processor = VideoProcessor(args.video, args.output)
    if not video_processor.initialize():
        return 1

    blade_tracker = BladeTracker(
        model_path=args.model,
        headless=args.headless,
        debug=args.debug,
        no_render=args.norender
    )

    if not blade_tracker.initialize():
        return 1

    # Run the tracking application
    try:
        blade_tracker.run(video_processor)
    finally:
        video_processor.release()

    return 0


if __name__ == "__main__":
    exit(main())
