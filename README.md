# OpenVLA-ROS2-Workspace

A comprehensive demonstration workspace for deploying OpenVLA (Vision-Language-Action) models in robotic applications, featuring multi-task support, performance benchmarking, and ROS 2 integration.

## 🌟 Key Features

- **Multi-Task Configuration**: Define and switch between manipulation tasks via YAML
- **Performance Benchmarking**: Track inference latency, FPS, memory usage, and success rates
- **Modular Architecture**: Reusable core components for model management
- **Three Demo Levels**:
  1. Basic inference with metrics
  2. Full ROS 2 integration
  3. Web-based control interface
- **Production-Ready**: Error handling, logging, and comprehensive testing

## 🎯 Distinguishing Features

Unlike basic VLA demos, this workspace provides:
- **Configurable task system** - No hardcoded prompts
- **Performance tracking** - Automated benchmarking suite
- **Better error handling** - Graceful degradation and logging
- **Extensible architecture** - Easy to add new tasks and demos
- **Documentation** - Detailed guides and architecture diagrams

## 📋 Prerequisites

- Ubuntu 22.04/24.04
- Python 3.10+
- NVIDIA GPU with 6GB+ VRAM
- CUDA 12.1+
- ROS 2 Jazzy (for Demo 2)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/OpenVLA-ROS2-Workspace.git
cd OpenVLA-ROS2-Workspace

# Run setup script
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh

# Activate environment
source venv/bin/activate
```

### 2. Download Test Images

```bash
mkdir -p data/test_images
# Add your test images here
```

### 3. Run Demo 1

```bash
python src/demos/demo1_basic_inference/run_inference.py \
  --image data/test_images/desk_scene.jpg \
  --task pick_and_place \
  --runs 10
```

## 📊 Performance Benchmarks

| GPU | VRAM | Quantization | Avg Latency | FPS |
|-----|------|--------------|-------------|-----|
| RTX 4090 | 24GB | 4-bit | ~165ms | ~6.0 |
| RTX 3090 | 24GB | 4-bit | ~250ms | ~4.0 |
| RTX 3060 | 12GB | 4-bit | ~400ms | ~2.5 |

*Benchmarks measured on Demo 1 with default configuration*

## 🏗️ Architecture

## 📚 Documentation

- [Architecture Details](docs/architecture.md)
- [Performance Benchmarks](docs/performance_benchmarks.md)
- [Adding Custom Tasks](docs/custom_tasks.md)
- [ROS 2 Integration Guide](src/demos/demo2_ros2_integration/README.md)

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest --cov=src tests/
```

## 📝 Configuration

All behavior is controlled via YAML files in `config/`:

- `model_config.yaml` - Model loading and quantization settings
- `tasks.yaml` - Task definitions and prompts
- `camera_config.yaml` - Camera and preprocessing settings

## 🤝 Contributing

This is an academic project. For improvements:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📄 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

- OpenVLA team for the base model
- ROS 2 community for robotics framework
- Hugging Face for model hosting

## 📧 Contact

[Your Name] - [Your Email]

Project Link: https://github.com/DeeptamBhar/OpenVLA-ROS2-Workspace