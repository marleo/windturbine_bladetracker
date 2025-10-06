import cv2
import numpy as np
from ultralytics import YOLO
import math
import argparse
import time
import itertools

# -------------------- Angle-Stable ID logic --------------------
def ang_diff(a, b):
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d

class AngleIDStabilizer:
    def __init__(self, n_blades=3, max_missing=10, alpha=0.7):
        self.n_blades = n_blades
        self.stable_id_map = {}
        self.stable_id_counter = 1
        self.last_known_angles = {}
        self.angular_velocities = {}
        self.missing_frames = {}
        self.max_missing = max_missing
        self.alpha = alpha

    def get_stable_ids(self, detections_with_ids, hub_cx, hub_cy, fps=None):
        current_tracker_ids = set()
        current_angles = {}
        for d in detections_with_ids:
            tracker_id, (det_cx, det_cy) = d['id'], d['centroid']
            current_tracker_ids.add(tracker_id)
            current_angles[tracker_id] = math.atan2(det_cy - hub_cy, det_cx - hub_cx)

        new_assignments = {}
        predicted_angles = {}
        for stable_id, last_angle in self.last_known_angles.items():
            if self.missing_frames[stable_id] < self.max_missing:
                pred_angle = last_angle + self.angular_velocities.get(stable_id, 0)
                pred_angle = (pred_angle + math.pi) % (2 * math.pi) - math.pi
                predicted_angles[stable_id] = pred_angle
                self.missing_frames[stable_id] += 1
            else:
                predicted_angles[stable_id] = last_angle

        unassigned_tracker_ids = list(current_tracker_ids)
        assigned_stable_ids = set()

        while unassigned_tracker_ids and len(assigned_stable_ids) < self.n_blades:
            best_match = None
            min_diff = float('inf')

            for stable_id, pred_angle in predicted_angles.items():
                if stable_id in assigned_stable_ids:
                    continue

                for tracker_id in unassigned_tracker_ids:
                    curr_angle = current_angles[tracker_id]
                    diff = abs(ang_diff(curr_angle, pred_angle))
                    if diff < min_diff:
                        min_diff = diff
                        best_match = (tracker_id, stable_id)

            if best_match:
                tracker_id, stable_id = best_match
                new_assignments[tracker_id] = stable_id
                self.angular_velocities[stable_id] = self.alpha * ang_diff(current_angles[tracker_id], self.last_known_angles.get(stable_id, current_angles[tracker_id])) + (1 - self.alpha) * self.angular_velocities.get(stable_id, 0)
                self.last_known_angles[stable_id] = current_angles[tracker_id]
                self.missing_frames[stable_id] = 0
                unassigned_tracker_ids.remove(tracker_id)
                assigned_stable_ids.add(stable_id)
                if stable_id in predicted_angles:
                    del predicted_angles[stable_id]
            else:
                break

        for tracker_id in unassigned_tracker_ids:
            if self.stable_id_counter <= self.n_blades:
                new_assignments[tracker_id] = self.stable_id_counter
                self.last_known_angles[self.stable_id_counter] = current_angles[tracker_id]
                self.angular_velocities[self.stable_id_counter] = 0
                self.missing_frames[self.stable_id_counter] = 0
                self.stable_id_counter += 1
            else:
                new_assignments[tracker_id] = -1

        return new_assignments

# -------------------- Helpers --------------------
def mask_centroid(mask_poly_xy):
    if mask_poly_xy is None or len(mask_poly_xy) == 0:
        return None
    x = mask_poly_xy[:, 0]
    y = mask_poly_xy[:, 1]
    a = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    if abs(a) < 1e-6:
        return float(np.mean(x)), float(np.mean(y))
    cx = np.sum((x + np.roll(x, -1)) * (x * np.roll(y, -1) - y * np.roll(x, -1))) / (3 * a)
    cy = np.sum((y + np.roll(y, -1)) * (x * np.roll(y, -1) - y * np.roll(x, -1))) / (3 * a)
    return float(cx), float(cy)

def fit_line_to_mask(mask_poly_xy):
    if mask_poly_xy is None or len(mask_poly_xy) < 2:
        return None
    points = mask_poly_xy.astype(np.float32)
    [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    return np.array([vx[0], vy[0]]), np.array([x0[0], y0[0]])

def compute_intersection(line1_dir, line1_pt, line2_dir, line2_pt):
    A = np.array([line1_dir, -line2_dir]).T
    b = line2_pt - line1_pt
    try:
        ts = np.linalg.solve(A, b)
        return line1_pt + ts[0] * line1_dir
    except:
        return None

# Parse arguments
parser = argparse.ArgumentParser(description='Wind Turbine Blade Tracking')
parser.add_argument('--headless', action='store_true', help='Run without UI')
parser.add_argument('--video', type=str, help='Path to input video file (optional)')
parser.add_argument('--output', type=str, help='Path to output video file (required when using --video)')
parser.add_argument('--debug', action='store_true', help='Show performance metrics overlay')
args = parser.parse_args()

# Validate arguments
if args.video and not args.output:
    parser.error("--output is required when using --video")
if args.headless and args.video:
    parser.error("--headless cannot be used with --video")

# Load the YOLO model
model = YOLO('yolov11s.pt')  # Assuming the model file is in the same directory

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

    # Process detections
    processing_start = time.time()
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
    processing_time = time.time() - processing_start

    # Hub estimation using line intersections
    hub_estimation_start = time.time()
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
    hub_estimation_time = time.time() - hub_estimation_start

    hub_cx, hub_cy = smoothed_hub_pos

    # ID stabilization (angle prediction)
    id_stabilization_start = time.time()
    stable_id_map = stabilizer.get_stable_ids(detections_with_ids, hub_cx, hub_cy)
    id_stabilization_time = time.time() - id_stabilization_start

    # Draw results
    rendering_start = time.time()
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

    # Display performance metrics (only in debug mode)
    if args.debug:
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                    FONT, FONT_SCALE, (0, 255, 0), THICKNESS, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Inference: {inference_time*1000:.1f}ms", (10, 60), 
                    FONT, FONT_SCALE, (255, 255, 0), THICKNESS, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Processing: {processing_time*1000:.1f}ms", (10, 90), 
                    FONT, FONT_SCALE, (255, 255, 0), THICKNESS, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Hub Est: {hub_estimation_time*1000:.1f}ms", (10, 120), 
                    FONT, FONT_SCALE, (255, 255, 0), THICKNESS, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"ID Stab: {id_stabilization_time*1000:.1f}ms", (10, 150), 
                    FONT, FONT_SCALE, (255, 255, 0), THICKNESS, cv2.LINE_AA)
        
        rendering_time = time.time() - rendering_start
        cv2.putText(annotated_frame, f"Rendering: {rendering_time*1000:.1f}ms", (10, 180), 
                    FONT, FONT_SCALE, (255, 255, 0), THICKNESS, cv2.LINE_AA)
    else:
        rendering_time = time.time() - rendering_start

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
