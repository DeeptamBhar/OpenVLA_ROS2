"""
Demo 3: Web-based control interface for OpenVLA
"""

from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from PIL import Image
import io
import base64
import sys
from pathlib import Path
import threading
import time

import math
import random
try:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from openvla_core.model_manager import ModelManager
    HAS_MODEL = True
except ImportError as e:
    print(f"Warning: Could not import ModelManager ({e}). Running in MOCK mode.")
    HAS_MODEL = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vla-demo-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
model_manager = None
current_task = "pick_and_place"
tasks = {}
processing = False
camera_feed = None


def load_tasks():
    """Load tasks from config"""
    import yaml
    with open('../../config/tasks.yaml', 'r') as f:
        return yaml.safe_load(f)


class MockModelManager:
    def load_model(self):
        return True

    def get_metrics(self):
        return {"inference_time": 0.5, "fps": 2.0, "memory_usage_mb": 450.0, "success_rate": 0.95}

    def infer(self, image, prompt, task=None):
        def noise(scale): return random.uniform(-scale, scale)
        latency = random.uniform(0.4, 0.9)
        
        # Output is [dx, dy, dz, droll, dpitch, dyaw, gripper]
        if task == 'pick_and_place':
            coords = [noise(0.01), noise(0.01), -0.05, 0, 0, 0, 0.0]
        elif task == 'push_to_target':
            coords = [0.08, 0.0, 0, 0, 0, 0, 1.0]
        elif task == 'reach_to_object':
            coords = [0.05, 0.05, -0.02, 0, 0, 0, 1.0]
        elif task == 'open_drawer':
            coords = [-0.05, 0.0, 0.02, 0, 0, 0, 0.0]
        else:
            angle = time.time() * 0.7
            coords = [0.10 * math.cos(angle), 0.10 * math.sin(angle), noise(0.03), noise(0.02), noise(0.02), noise(0.05), 1.0]

        action_str = "Coordinates: [" + ", ".join(f"{v:.4f}" for v in coords) + "]"
        return {"action": action_str, "inference_time": latency}

def initialize_model():
    """Initialize VLA model"""
    global model_manager
    if HAS_MODEL:
        print("Loading VLA model...")
        model_manager = ModelManager('../../config/model_config.yaml')
        if not model_manager.load_model():
            print("Failed to load model!")
            return False
        print("Model loaded successfully!")
        return True
    else:
        print("Initializing Mock VLA model...")
        model_manager = MockModelManager()
        return True


# ── Color-based object detection ─────────────────────────────────────────────
# Maps keyword lists → HSV ranges (H: 0-180 like OpenCV, S/V: 0-255)
_COLOR_RULES = [
    (['green', 'plant', 'leaf', 'grass', 'tree'],   35,  85,  60,  40),
    (['red',   'tomato', 'apple', 'strawberry'],      0,  10,  80,  60),
    (['red',   'tomato', 'apple', 'strawberry'],    160, 180,  80,  60),  # red wraps hue
    (['blue',  'bin',   'bowl'],                     95, 130,  60,  40),
    (['yellow','banana','lemon'],                     22,  35,  80,  80),
    (['orange','carrot'],                             10,  22,  80,  80),
    (['white', 'paper', 'plate'],                     0, 180,   0, 200),
    (['purple','violet'],                            130, 160,  50,  40),
    (['pink',  'rose'],                              155, 175,  40, 150),
    (['brown', 'wood',  'table'],                     10,  20,  40,  40),
]

def _rgb_to_hsv_arr(arr: np.ndarray) -> np.ndarray:
    """Convert H×W×3 uint8 RGB → H×W×3 float HSV (H:0-180, S:0-255, V:0-255)."""
    r = arr[:,:,0] / 255.0
    g = arr[:,:,1] / 255.0
    b = arr[:,:,2] / 255.0
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    diff = cmax - cmin + 1e-9

    h = np.zeros_like(r)
    mr = (cmax == r) & (diff > 1e-8)
    mg = (cmax == g) & (diff > 1e-8)
    mb = (cmax == b) & (diff > 1e-8)
    h[mr] = (60 * ((g[mr] - b[mr]) / diff[mr])) % 360
    h[mg] = (60 * ((b[mg] - r[mg]) / diff[mg])) + 120
    h[mb] = (60 * ((r[mb] - g[mb]) / diff[mb])) + 240
    h /= 2.0          # 0-180 scale

    s = np.where(cmax > 0, (diff / (cmax + 1e-9)) * 255, 0)
    v = cmax * 255
    return np.stack([h, s, v], axis=-1)


def detect_object_location(image_b64: str, object_name: str):
    """
    Detect pixel centroid of an object by matching its colour keyword.
    Returns {x, y, pixel_count} normalised to [0,1], or None if not detectable.
    """
    if not object_name or ',' not in image_b64:
        return None
    obj = object_name.lower()

    # Decode image
    _, data = image_b64.split(',', 1)
    img = Image.open(io.BytesIO(base64.b64decode(data))).convert('RGB')
    arr = np.array(img, dtype=np.uint8)
    hsv = _rgb_to_hsv_arr(arr)
    H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    # Special case: very dark / black objects
    if any(w in obj for w in ['black', 'dark']):
        mask = (V < 60) & (S < 60)
        if mask.any():
            ys, xs = np.where(mask)
            return {'x': float(xs.mean()) / arr.shape[1],
                    'y': float(ys.mean()) / arr.shape[0],
                    'pixel_count': int(mask.sum())}
        return None

    best_mask, best_count = None, 0
    for keywords, h_lo, h_hi, s_min, v_min in _COLOR_RULES:
        if not any(kw in obj for kw in keywords):
            continue
        m = (H >= h_lo) & (H <= h_hi) & (S >= s_min) & (V >= v_min)
        cnt = int(m.sum())
        if cnt > best_count:
            best_count = cnt
            best_mask  = m

    if best_mask is None or best_count < 50:
        return None

    ys, xs = np.where(best_mask)
    return {
        'x':           float(xs.mean()) / arr.shape[1],
        'y':           float(ys.mean()) / arr.shape[0],
        'pixel_count': best_count,
    }


def process_image(image_data, prompt, object_name=None):
    """Process image and return results"""
    global processing
    
    if processing:
        return {'error': 'Already processing'}
    
    processing = True
    
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        
        # Run inference
        result = model_manager.infer(image, prompt, task=current_task)
        
        if result is None:
            return {'error': 'Inference failed'}
        
        # Get metrics
        metrics = model_manager.get_metrics()
        
        target_pixel = detect_object_location(image_data, object_name) if object_name else None
        
        return {
            'success': True,
            'action': result['action'],
            'inference_time': result['inference_time'],
            'metrics': metrics,
            'task': current_task,
            'target_pixel': target_pixel
        }
    
    except Exception as e:
        return {'error': str(e)}
    
    finally:
        processing = False


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', tasks=tasks)


@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    """Get available tasks"""
    return jsonify(tasks)


@app.route('/api/task', methods=['POST'])
def set_task():
    """Set current task"""
    global current_task
    data = request.json
    task_name = data.get('task')
    
    if task_name in tasks['tasks']:
        current_task = task_name
        return jsonify({'success': True, 'task': current_task})
    else:
        return jsonify({'success': False, 'error': 'Invalid task'}), 400


@app.route('/api/infer', methods=['POST'])
def infer():
    """Run inference on uploaded image"""
    data = request.json
    image_data = data.get('image')
    prompt = data.get('prompt')
    object_name = data.get('object', '')
    
    if not image_data or not prompt:
        return jsonify({'error': 'Missing image or prompt'}), 400
    
    result = process_image(image_data, prompt, object_name)
    return jsonify(result)


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get current metrics"""
    if model_manager is None:
        return jsonify({'error': 'Model not loaded'}), 400
    
    metrics = model_manager.get_metrics()
    return jsonify(metrics)


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('status', {'message': 'Connected to VLA server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')


@socketio.on('process_image')
def handle_process_image(data):
    """Handle real-time image processing"""
    image_data = data.get('image')
    prompt = data.get('prompt', tasks['tasks'][current_task]['prompt'])
    
    object_name = data.get('object', 'cup')
    
    # Format prompt
    prompt = prompt.format(
        object=object_name,
        container=data.get('container', 'bin'),
        target=data.get('target', 'left'),
        drawer=data.get('drawer', 'top')
    )
    
    result = process_image(image_data, prompt, object_name)
    
    emit('inference_result', result)


if __name__ == '__main__':
    print("Initializing OpenVLA Web Interface...")
    
    # Load tasks
    tasks = load_tasks()
    print(f"Loaded {len(tasks['tasks'])} tasks")
    
    # Initialize model
    if not initialize_model():
        print("Failed to initialize model. Exiting.")
        sys.exit(1)
    
    print("\nStarting web server...")
    print("Open http://localhost:5000 in your browser")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    