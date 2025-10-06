"""
Wind Turbine Blade Tracking Application

This application uses YOLO segmentation models to track wind turbine blades in real-time
from webcam feed or process recorded videos. It features advanced tracking algorithms
including dynamic hub detection, angle-based ID assignment, and prediction for stable
blade identification through occlusions.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import math
import argparse
import time
import itertools
from typing import List, Dict, Tuple, Optional, Any

# =============================================================================
# ANGLE-BASED ID STABILIZATION
# =============================================================================

class AngleIDStabilizer:
    """
    Stabilizes blade IDs using angular position prediction and history.

    This class maintains consistent blade identification even when blades
    temporarily disappear from view by predicting their positions based on
    angular velocity and matching reappearing blades to predicted locations.
    """

    def __init__(self, n_blades: int = 3, max_missing: int = 10, alpha: float = 0.7):
        """
        Initialize the stabilizer.

        Args:
            n_blades: Expected number of blades in the turbine
            max_missing: Maximum frames a blade can be missing before forgetting it
            alpha: Smoothing factor for angular velocity (0-1, higher = more smoothing)
        """
        self.n_blades = n_blades
        self.max_missing = max_missing
        self.alpha = alpha

        # Tracking state
        self.stable_id_counter = 1
        self.stable_id_map = {}  # tracker_id -> stable_id
        self.last_known_angles = {}  # stable_id -> last angle
        self.angular_velocities = {}  # stable_id -> angular velocity
        self.missing_frames = {}  # stable_id -> frames missing

    def get_stable_ids(self, detections_with_ids: List[Dict], hub_cx: float, hub_cy: float, fps: Optional[float] = None) -> Dict[int, int]:
        """
        Assign stable IDs to detections based on angular position and prediction.

        Args:
            detections_with_ids: List of detection dicts with 'id' and 'centroid' keys
            hub_cx, hub_cy: Current hub position coordinates
            fps: Frames per second (optional, for future use)

        Returns:
            Dictionary mapping tracker IDs to stable blade IDs
        """
        # Extract current angles for all detected blades
        current_tracker_ids = set()
        current_angles = {}

        for detection in detections_with_ids:
            tracker_id = detection['id']
            centroid_x, centroid_y = detection['centroid']
            current_tracker_ids.add(tracker_id)
            current_angles[tracker_id] = math.atan2(centroid_y - hub_cy, centroid_x - hub_cx)

        # Predict positions for missing blades
        predicted_angles = self._predict_missing_blade_angles()

        # Match current detections to predicted positions
        assignments = self._match_detections_to_predictions(
            current_tracker_ids, current_angles, predicted_angles
        )

        # Assign new IDs to unmatched detections
        assignments = self._assign_new_ids_to_unmatched(
            current_tracker_ids, current_angles, assignments
        )

        return assignments

    def _predict_missing_blade_angles(self) -> Dict[int, float]:
        """Predict angular positions for blades that are currently missing."""
        predicted_angles = {}

        for stable_id, last_angle in self.last_known_angles.items():
            if self.missing_frames[stable_id] < self.max_missing:
                # Predict new angle based on angular velocity
                predicted_angle = last_angle + self.angular_velocities.get(stable_id, 0)
                # Normalize to [-π, π]
                predicted_angle = (predicted_angle + math.pi) % (2 * math.pi) - math.pi
                predicted_angles[stable_id] = predicted_angle
                self.missing_frames[stable_id] += 1
            else:
                # Blade has been missing too long, keep last known position
                predicted_angles[stable_id] = last_angle

        return predicted_angles

    def _match_detections_to_predictions(self, current_tracker_ids: set, current_angles: Dict[int, float],
                                       predicted_angles: Dict[int, float]) -> Dict[int, int]:
        """Match current detections to predicted blade positions."""
        assignments = {}
        unassigned_tracker_ids = list(current_tracker_ids)
        assigned_stable_ids = set()

        # Greedily assign best matches
        while unassigned_tracker_ids and len(assigned_stable_ids) < self.n_blades:
            best_match = None
            min_angular_diff = float('inf')

            for stable_id, predicted_angle in predicted_angles.items():
                if stable_id in assigned_stable_ids:
                    continue

                for tracker_id in unassigned_tracker_ids:
                    current_angle = current_angles[tracker_id]
                    angular_diff = abs(ang_diff(current_angle, predicted_angle))

                    if angular_diff < min_angular_diff:
                        min_angular_diff = angular_diff
                        best_match = (tracker_id, stable_id)

            if best_match:
                tracker_id, stable_id = best_match
                assignments[tracker_id] = stable_id

                # Update tracking state
                self._update_blade_state(stable_id, current_angles[tracker_id])

                unassigned_tracker_ids.remove(tracker_id)
                assigned_stable_ids.add(stable_id)

                # Remove from predictions once assigned
                if stable_id in predicted_angles:
                    del predicted_angles[stable_id]
            else:
                break

        return assignments

    def _assign_new_ids_to_unmatched(self, current_tracker_ids: set, current_angles: Dict[int, float],
                                    assignments: Dict[int, int]) -> Dict[int, int]:
        """Assign new stable IDs to detections that couldn't be matched to predictions."""
        for tracker_id in current_tracker_ids:
            if tracker_id not in assignments:
                if self.stable_id_counter <= self.n_blades:
                    stable_id = self.stable_id_counter
                    assignments[tracker_id] = stable_id
                    self._initialize_blade_state(stable_id, current_angles[tracker_id])
                    self.stable_id_counter += 1
                else:
                    # Too many blades detected, mark as invalid
                    assignments[tracker_id] = -1

        return assignments

    def _update_blade_state(self, stable_id: int, current_angle: float):
        """Update the state of a matched blade."""
        # Update angular velocity with exponential smoothing
        last_angle = self.last_known_angles.get(stable_id, current_angle)
        angle_change = ang_diff(current_angle, last_angle)
        current_velocity = self.angular_velocities.get(stable_id, 0)

        self.angular_velocities[stable_id] = (self.alpha * angle_change +
                                            (1 - self.alpha) * current_velocity)
        self.last_known_angles[stable_id] = current_angle
        self.missing_frames[stable_id] = 0

    def _initialize_blade_state(self, stable_id: int, initial_angle: float):
        """Initialize state for a newly detected blade."""
        self.last_known_angles[stable_id] = initial_angle
        self.angular_velocities[stable_id] = 0
        self.missing_frames[stable_id] = 0

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ang_diff(a: float, b: float) -> float:
    """
    Calculate the smallest angular difference between two angles in radians.

    Args:
        a: First angle in radians
        b: Second angle in radians

    Returns:
        Angular difference normalized to [-π, π]
    """
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d


def mask_centroid(mask_poly_xy: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Calculate the centroid of a polygon using the shoelace formula.

    Args:
        mask_poly_xy: Nx2 array of polygon vertices

    Returns:
        (cx, cy) centroid coordinates, or None if calculation fails
    """
    if mask_poly_xy is None or len(mask_poly_xy) == 0:
        return None

    x = mask_poly_xy[:, 0]
    y = mask_poly_xy[:, 1]
    a = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))

    if abs(a) < 1e-6:
        # Degenerate polygon, return mean
        return float(np.mean(x)), float(np.mean(y))

    # Use shoelace formula for centroid
    cx = np.sum((x + np.roll(x, -1)) * (x * np.roll(y, -1) - y * np.roll(x, -1))) / (3 * a)
    cy = np.sum((y + np.roll(y, -1)) * (x * np.roll(y, -1) - y * np.roll(x, -1))) / (3 * a)
    return float(cx), float(cy)


def fit_line_to_mask(mask_poly_xy: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Fit a line to a polygon mask using OpenCV's fitLine function.

    Args:
        mask_poly_xy: Nx2 array of polygon vertices

    Returns:
        (direction_vector, point_on_line) or None if fitting fails
    """
    if mask_poly_xy is None or len(mask_poly_xy) < 2:
        return None

    points = mask_poly_xy.astype(np.float32)
    [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    return np.array([vx[0], vy[0]]), np.array([x0[0], y0[0]])


def compute_intersection(line1_dir: np.ndarray, line1_pt: np.ndarray,
                        line2_dir: np.ndarray, line2_pt: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute intersection point of two lines.

    Args:
        line1_dir: Direction vector of first line
        line1_pt: Point on first line
        line2_dir: Direction vector of second line
        line2_pt: Point on second line

    Returns:
        Intersection point or None if lines are parallel
    """
    A = np.array([line1_dir, -line2_dir]).T
    b = line2_pt - line1_pt
    try:
        ts = np.linalg.solve(A, b)
        return line1_pt + ts[0] * line1_dir
    except np.linalg.LinAlgError:
        return None

# Parse arguments
parser = argparse.ArgumentParser(description='Wind Turbine Blade Tracking')
parser.add_argument('--headless', action='store_true', help='Run without UI')
parser.add_argument('--video', type=str, help='Path to input video file (optional)')
parser.add_argument('--output', type=str, help='Path to output video file (required when using --video)')
args = parser.parse_args()

# Validate arguments
if args.video and not args.output:
    parser.error("--output is required when using --video")
if args.headless and args.video:
    parser.error("--headless cannot be used with --video")

# Load the YOLO model
model = YOLO('yolov11n_full_2609.pt')  # Assuming the model file is in the same directory

# Configuration
N_BLADES = 3
IMG_SIZE = 640
CONF_THRES = 0.15
IOU_THRES = 0.5
TRACKER_CFG = "bytetrack.yaml"
HUB_SMOOTHING_ALPHA = 0.1
BLADE_COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
THICKNESS = 1

# Input setup (webcam or video)
if args.video:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {args.video}")
        exit()
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (W, H))
    print(f"Processing video: {args.video}")
    print(f"Output: {args.output}")
    print(f"Resolution: {W}x{H}, FPS: {fps}, Frames: {total_frames}")
else:
    # Webcam setup
    cap = cv2.VideoCapture(1)  # 0 for default webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize stabilizer and hub position
stabilizer = AngleIDStabilizer(n_blades=N_BLADES)
smoothed_hub_pos = np.array([W/2, H/2])

# Main loop
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    height, width = frame.shape[:2]

    # Measure inference time
    inference_start = time.time()
    
    # Run inference with tracking
    results = model.track(
        frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        tracker=TRACKER_CFG,
        persist=True,
        verbose=False
    )
    
    inference_time = time.time() - inference_start
    fps = 1.0 / inference_time if inference_time > 0 else 0

    detections_with_ids = []
    r0 = results[0]
    if r0.boxes and r0.boxes.id is not None:
        track_ids = r0.boxes.id.int().cpu().tolist()
        if r0.masks:
            for i, poly in enumerate(r0.masks.xy):
                c = mask_centroid(np.array(poly, dtype=np.float32))
                if c:
                    detections_with_ids.append({
                        'id': track_ids[i],
                        'centroid': c,
                        'poly': np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                    })
        else:
            for i, (x1, y1, x2, y2) in enumerate(r0.boxes.xyxy.cpu().numpy()):
                detections_with_ids.append({
                    'id': track_ids[i],
                    'centroid': ((x1+x2)/2, (y1+y2)/2),
                    'poly': None
                })

    # Hub estimation using line intersections
    mask_lines = []
    for d in detections_with_ids:
        if d['poly'] is not None:
            poly = d['poly'].reshape(-1, 2).astype(np.float32)
            line_dir, line_pt = fit_line_to_mask(poly)
            if line_dir is not None:
                mask_lines.append((line_dir, line_pt))

    if len(mask_lines) >= 2:
        intersections = []
        for i, j in itertools.combinations(range(len(mask_lines)), 2):
            line1_dir, line1_pt = mask_lines[i]
            line2_dir, line2_pt = mask_lines[j]
            inter = compute_intersection(line1_dir, line1_pt, line2_dir, line2_pt)
            if inter is not None and 0 <= inter[0] < width and 0 <= inter[1] < height:
                intersections.append(inter)
        if intersections:
            new_hub = np.mean(intersections, axis=0)
            smoothed_hub_pos = (HUB_SMOOTHING_ALPHA * new_hub + 
                              (1 - HUB_SMOOTHING_ALPHA) * smoothed_hub_pos)

    hub_cx, hub_cy = smoothed_hub_pos

    stable_id_map = stabilizer.get_stable_ids(detections_with_ids, hub_cx, hub_cy)

    # Draw results
    annotated_frame = frame.copy()
    for d in detections_with_ids:
        stable_id = stable_id_map.get(d['id'])
        if stable_id and stable_id > 0:
            px, py = d['centroid']
            color = BLADE_COLORS[(stable_id - 1) % len(BLADE_COLORS)]
            if d['poly'] is not None:
                overlay = annotated_frame.copy()
                cv2.fillPoly(overlay, [d['poly']], color)
                cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0, annotated_frame)
            cv2.putText(annotated_frame, f"Blade {stable_id}", (int(px)+6,int(py)-6),
                        FONT, FONT_SCALE, (0,0,0), THICKNESS+2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Blade {stable_id}", (int(px)+6,int(py)-6),
                        FONT, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)
            cv2.circle(annotated_frame, (int(px), int(py)), 4, color, -1)

    cv2.circle(annotated_frame, (int(hub_cx), int(hub_cy)), 5, (255,255,255), -1)
    cv2.circle(annotated_frame, (int(hub_cx), int(hub_cy)), 9, (0,0,0), 2)

    # Display FPS
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                FONT, FONT_SCALE, (0, 255, 0), THICKNESS, cv2.LINE_AA)

    if args.video:
        # Write frame to output video
        out.write(annotated_frame)
        print(f"\rProcessing frame {frame_count}/{total_frames} (FPS: {fps:.1f})", end="", flush=True)
    elif not args.headless:
        cv2.imshow('Blade Tracking', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        time.sleep(0.033)  # ~30 fps in headless mode

cap.release()
if args.video:
    out.release()
    print(f"\nVideo processing completed. Output saved to: {args.output}")
cv2.destroyAllWindows()
