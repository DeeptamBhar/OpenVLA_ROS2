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

sys.path.append(str(Path(__file__).parent.parent.parent))
from openvla_core.model_manager import ModelManager

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
    with open('../../../config/tasks.yaml', 'r') as f:
        return yaml.safe_load(f)


def initialize_model():
    """Initialize VLA model"""
    global model_manager
    print("Loading VLA model...")
    model_manager = ModelManager('../../../config/model_config.yaml')
    if not model_manager.load_model():
        print("Failed to load model!")
        return False
    print("Model loaded successfully!")
    return True


def process_image(image_data, prompt):
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
        
        return {
            'success': True,
            'action': result['action'],
            'inference_time': result['inference_time'],
            'metrics': metrics
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
    
    if not image_data or not prompt:
        return jsonify({'error': 'Missing image or prompt'}), 400
    
    result = process_image(image_data, prompt)
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
    
    # Format prompt
    prompt = prompt.format(
        object=data.get('object', 'cup'),
        container=data.get('container', 'bin'),
        target=data.get('target', 'left'),
        drawer=data.get('drawer', 'top')
    )
    
    result = process_image(image_data, prompt)
    
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
    