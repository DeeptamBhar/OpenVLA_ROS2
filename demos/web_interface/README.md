# Demo 2: Web-based Control Interface

Modern web interface for controlling OpenVLA with real-time visualization.

## Features

- Clean, modern web UI with drag-and-drop image upload
- **Interactive canvas visualization** of robotic predictions (arrows, targets, and tool paths)
- **Task-specific graphics** (e.g., lift arrows for pick_and_place, crosshairs for reaching)
- Built-in **color-based object detection** to point directly at specified items in the image
- Real-time task switching and live performance metrics
- WebSocket communication and responsive design

## Quick Start

```bash
cd demos/web_interface

# Install dependencies
pip install flask flask-socketio python-socketio

# Run server
python app.py
```

Then open http://localhost:5000 in your browser.

## Usage

1. Upload an image (click upload area or drag & drop)
2. Select a task from dropdown
3. Customize prompt parameters (object, container, etc.)
4. Click "Run Inference"
5. View predicted actions and performance metrics

## API Endpoints

- `GET /` - Main web interface
- `GET /api/tasks` - Get available tasks
- `POST /api/task` - Switch current task
- `POST /api/infer` - Run inference
- `GET /api/metrics` - Get performance metrics

## WebSocket Events

- `process_image` - Submit image for processing
- `inference_result` - Receive inference results
- `status` - Status updates