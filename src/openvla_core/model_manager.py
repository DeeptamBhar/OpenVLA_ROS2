"""
OpenVLA Model Manager
Handles model loading, caching, and quantization with performance monitoring
"""

import torch
import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from dataclasses import dataclass
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Track model performance metrics"""
    load_time: float = 0.0
    inference_times: list = None
    memory_usage_mb: float = 0.0
    successful_inferences: int = 0
    failed_inferences: int = 0
    
    def __post_init__(self):
        if self.inference_times is None:
            self.inference_times = []
    
    def avg_inference_time(self) -> float:
        return sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0.0
    
    def fps(self) -> float:
        avg_time = self.avg_inference_time()
        return 1.0 / avg_time if avg_time > 0 else 0.0


class ModelManager:
    """
    Manages OpenVLA model lifecycle with performance tracking
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config = self._load_config(config_path)
        self.model = None
        self.processor = None
        self.metrics = ModelMetrics()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initialized ModelManager on device: {self.device}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Create quantization configuration"""
        if not self.config['model']['quantization']['enabled']:
            return None
        
        quant_config = self.config['model']['quantization']
        
        return BitsAndBytesConfig(
            load_in_4bit=(quant_config['bits'] == 4),
            load_in_8bit=(quant_config['bits'] == 8),
            bnb_4bit_compute_dtype=getattr(torch, quant_config['compute_dtype']),
            bnb_4bit_quant_type=quant_config['method'],
            bnb_4bit_use_double_quant=quant_config['double_quant']
        )
    
    def load_model(self) -> bool:
        """
        Load the OpenVLA model with quantization
        Returns True if successful, False otherwise
        """
        try:
            start_time = time.time()
            model_name = self.config['model']['name']
            
            logger.info(f"Loading model: {model_name}")
            logger.info(f"Quantization: {self.config['model']['quantization']['enabled']}")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Get quantization config
            quantization_config = self._get_quantization_config()
            
            # Load model
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=getattr(torch, self.config['model']['inference']['torch_dtype']),
                device_map="auto",
                trust_remote_code=True,
                cache_dir=self.config['model']['cache_dir']
            )
            
            # Compile model if enabled
            if self.config['model']['inference']['compile']:
                logger.info("Compiling model with torch.compile()...")
                compile_mode = self.config['model']['inference']['compile_mode']
                self.model = torch.compile(self.model, mode=compile_mode)
            
            # Track metrics
            self.metrics.load_time = time.time() - start_time
            self.metrics.memory_usage_mb = torch.cuda.memory_allocated() / 1024**2
            
            logger.info(f"Model loaded successfully in {self.metrics.load_time:.2f}s")
            logger.info(f"Memory usage: {self.metrics.memory_usage_mb:.2f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def infer(self, image, prompt: str, task: str = "default") -> Optional[Dict[str, Any]]:
        """
        Run inference on image with given prompt
        
        Args:
            image: PIL Image or numpy array
            prompt: Text prompt for the task
            task: Task type (used for post-processing)
            
        Returns:
            Dictionary with action prediction and metadata
        """
        if self.model is None or self.processor is None:
            logger.error("Model not loaded. Call load_model() first.")
            return None
        
        try:
            start_time = time.time()
            
            # Prepare inputs
            inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=self.config['model']['performance']['temperature'],
                    do_sample=False
                )
            
            # Decode action
            action = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Track metrics
            inference_time = time.time() - start_time
            self.metrics.inference_times.append(inference_time)
            self.metrics.successful_inferences += 1
            
            result = {
                'action': action,
                'inference_time': inference_time,
                'prompt': prompt,
                'task': task,
                'timestamp': time.time()
            }
            
            logger.info(f"Inference completed in {inference_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            self.metrics.failed_inferences += 1
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return current performance metrics"""
        return {
            'load_time': self.metrics.load_time,
            'avg_inference_time': self.metrics.avg_inference_time(),
            'fps': self.metrics.fps(),
            'memory_usage_mb': self.metrics.memory_usage_mb,
            'total_inferences': self.metrics.successful_inferences + self.metrics.failed_inferences,
            'successful': self.metrics.successful_inferences,
            'failed': self.metrics.failed_inferences,
            'success_rate': (
                self.metrics.successful_inferences / 
                (self.metrics.successful_inferences + self.metrics.failed_inferences)
                if (self.metrics.successful_inferences + self.metrics.failed_inferences) > 0 
                else 0.0
            )
        }
    
    def print_metrics(self):
        """Print formatted metrics"""
        metrics = self.get_metrics()
        print("\n" + "="*50)
        print("MODEL PERFORMANCE METRICS")
        print("="*50)
        print(f"Load Time:           {metrics['load_time']:.2f}s")
        print(f"Avg Inference Time:  {metrics['avg_inference_time']*1000:.1f}ms")
        print(f"FPS:                 {metrics['fps']:.2f}")
        print(f"Memory Usage:        {metrics['memory_usage_mb']:.2f} MB")
        print(f"Total Inferences:    {metrics['total_inferences']}")
        print(f"Success Rate:        {metrics['success_rate']*100:.1f}%")
        print("="*50 + "\n")