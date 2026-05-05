# Demo 3: Web-based Control Interface

Modern web interface for controlling OpenVLA with real-time visualization.

## Features

- 🌐 Clean, modern web UI
- 📸 Drag-and-drop image upload
- 🔄 Real-time task switching
- 📊 Live performance metrics
- ⚡ WebSocket communication
- 📱 Responsive design

## Quick Start

```bash
cd src/demos/demo3_web_interface

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