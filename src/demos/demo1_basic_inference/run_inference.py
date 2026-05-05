"""
Demo 1: Basic OpenVLA Inference with Performance Benchmarking
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from openvla_core.model_manager import ModelManager
from PIL import Image
import yaml
import argparse
import time


def load_tasks(tasks_config_path: str = "config/tasks.yaml"):
    """Load task configurations"""
    with open(tasks_config_path, 'r') as f:
        return yaml.safe_load(f)


def run_demo(image_path: str, task_name: str = "pick_and_place", num_runs: int = 5):
    """
    Run basic inference demo with benchmarking
    
    Args:
        image_path: Path to input image
        task_name: Name of task from tasks.yaml
        num_runs: Number of inference runs for benchmarking
    """
    print(f"\n{'='*60}")
    print(f"OpenVLA Basic Inference Demo")
    print(f"{'='*60}\n")
    
    # Load configurations
    print("Loading configurations...")
    tasks_config = load_tasks()
    
    if task_name not in tasks_config['tasks']:
        print(f"Error: Task '{task_name}' not found in tasks.yaml")
        print(f"Available tasks: {list(tasks_config['tasks'].keys())}")
        return
    
    task = tasks_config['tasks'][task_name]
    
    # Initialize model manager
    print("Initializing Model Manager...")
    manager = ModelManager()
    
    # Load model
    print("\nLoading OpenVLA model...")
    if not manager.load_model():
        print("Failed to load model. Exiting.")
        return
    
    # Load image
    print(f"\nLoading image: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"Image size: {image.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Prepare prompt
    # Example: Replace placeholders
    prompt = task['prompt'].format(object="cup", container="drawer")
    print(f"\nTask: {task_name}")
    print(f"Prompt: {prompt}")
    print(f"Description: {task['description']}")
    print(f"Action dimensions: {task['action_dims']}")
    
    # Run inference multiple times for benchmarking
    print(f"\n{'='*60}")
    print(f"Running {num_runs} inference iterations...")
    print(f"{'='*60}\n")
    
    for i in range(num_runs):
        print(f"Iteration {i+1}/{num_runs}...", end=" ")
        result = manager.infer(image, prompt, task=task_name)
        
        if result:
            print(f"✓ ({result['inference_time']*1000:.1f}ms)")
            if i == 0:  # Print first result
                print(f"  Predicted action: {result['action']}")
        else:
            print("✗ Failed")
    
    # Print final metrics
    print()
    manager.print_metrics()
    
    # Save metrics to file
    metrics = manager.get_metrics()
    metrics_file = f"metrics_{task_name}_{int(time.time())}.yaml"
    with open(metrics_file, 'w') as f:
        yaml.dump(metrics, f)
    print(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenVLA Basic Inference Demo")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--task", type=str, default="pick_and_place", help="Task name from tasks.yaml")
    parser.add_argument("--runs", type=int, default=5, help="Number of inference runs")
    
    args = parser.parse_args()
    
    run_demo(args.image, args.task, args.runs)