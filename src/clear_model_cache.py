import os
import shutil
import time
import torch

def clear_model_cache():
    """
    Clear model caches to ensure fresh loading with CPU settings.
    """
    # 1. Set environment variables to force CPU usage
    os.environ["PYTORCH_DEVICE"] = "cpu"
    os.environ["PYTORCH_MPS_DEVICE"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 2. Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 3. Clear Hugging Face Transformers cache directories
    cache_paths = [
        "./models",  # From your code
        os.path.expanduser("~/.cache/huggingface/hub"),  # Default HF cache
        os.path.expanduser("~/.cache/torch"),  # PyTorch cache
    ]
    
    for cache_path in cache_paths:
        if os.path.exists(cache_path):
            print(f"Clearing cache at: {cache_path}")
            try:
                # For directories like the HF cache, just remove specific model directories
                if "huggingface" in cache_path:
                    for model_dir in ["clip", "e5", "multilingual"]:
                        model_path = os.path.join(cache_path, model_dir)
                        if os.path.exists(model_path):
                            shutil.rmtree(model_path)
                            print(f"Removed {model_path}")
                # For project-specific caches, remove them entirely
                elif cache_path == "./models":
                    shutil.rmtree(cache_path)
                    print(f"Removed {cache_path}")
            except Exception as e:
                print(f"Error clearing cache at {cache_path}: {e}")
    
    print("Model caches cleared successfully!")

if __name__ == "__main__":
    clear_model_cache()
    
    # Wait a moment for any file operations to complete
    time.sleep(1)
    
    print("\nNow running evaluator with CPU-only settings...")
    print("You can now run the evaluator with:")
    print("  python src/evaluation/approach_evaluator.py --approach opensource --num-questions 1")