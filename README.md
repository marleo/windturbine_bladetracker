# Wind Turbine Blade Tracking

This application uses a pretrained YOLOv11 segmentation model to perform live inference on a webcam feed for tracking wind turbine blades. It features advanced tracking algorithms including dynamic hub detection, angular velocity prediction, and stable ID assignment that maintains blade identities through occlusions and reappearances.

## Features

- Real-time blade detection and segmentation with mask visualization
- Advanced multi-object tracking using YOLO's ByteTrack integration
- Dynamic hub position estimation using blade mask line intersections
- Angle-based ID assignment with prediction for stable blade identification
- Handles partial blade visibility and temporary tracking loss with angular velocity prediction
- Real-time FPS counter for inference performance monitoring
- Video processing mode for offline analysis with output video generation
- Headless mode for deployment without UI
- Optimized for lightweight execution on Jetson devices

## Requirements

- Python 3.8+
- Webcam (for live tracking) or video file (for offline processing)
- Pretrained YOLO11n-seg model file (see Setup section)

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd jetson_bladetracker
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download or place your pretrained YOLO model file in the project directory:
   - The model file should be named according to your training (e.g., `yolov11n_full_2609.pt`)
   - Update the `model_path` variable in `main.py` if using a different filename

## Usage

Run the application:
```bash
python main.py
```

For headless mode (no UI):
```bash
python main.py --headless
```

For video processing:
```bash
python main.py --video input_video.mp4 --output output_video.mp4
```

- In normal mode, press 'q' to quit the application.
- The application will display the webcam feed with detected blades marked with segmentation masks, circles, assigned IDs, and real-time FPS counter.
- Video processing mode will process the entire video and save the annotated output without displaying a UI.

## Configuration

- The hub position is dynamically estimated from blade mask intersections
- Tracking uses YOLO's built-in ByteTrack with configurable parameters
- Inference confidence threshold is set to 0.15; modify CONF_THRES as needed
- Angular velocity prediction maintains IDs through occlusions up to 10 frames

## Project Structure

```
jetson_bladetracker/
├── main.py                 # Main application script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
└── yolov11n_full_2609.pt  # YOLO model (not included in repo)
```

## Important Notes

- **Model File**: The YOLO model file is excluded from version control due to its size. You must obtain and place it in the project directory manually.
- **Output Files**: Generated video files are also excluded from version control.
- **Virtual Environment**: Consider using a virtual environment to avoid dependency conflicts.