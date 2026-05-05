# Demo 1: Basic Inference with Benchmarking

This demo showcases OpenVLA's inference capabilities with comprehensive performance benchmarking.

## Features

- ✅ Multi-task support via configuration files
- ✅ Performance metrics tracking (latency, FPS, success rate)
- ✅ Automatic model quantization
- ✅ Metrics export to YAML

## Quick Start

```bash
# From repository root
python src/demos/demo1_basic_inference/run_inference.py \
  --image data/test_images/desk_scene.jpg \
  --task pick_and_place \
  --runs 10
```

## Configuration

Tasks are defined in `config/tasks.yaml`. Add custom tasks:

```yaml
my_custom_task:
  prompt: "Do something with {object}"
  description: "My custom manipulation task"
  action_dims: 7
```

## Output

The demo generates:
1. Console output with real-time metrics
2. `metrics_<task>_<timestamp>.yaml` file with detailed performance data