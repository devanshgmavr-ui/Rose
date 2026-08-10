#!/usr/bin/env python3
"""Model download script for the local agent."""

import sys
import os
import argparse
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Available models (GGUF format from HuggingFace)
MODELS = {
    "qwen2.5-coder-14b-q4": {
        "name": "Qwen2.5-Coder-14B-Instruct (Q4_K_M)",
        "url": "https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF/resolve/main/qwen2.5-coder-14b-instruct-q4_k_m.gguf",
        "filename": "qwen2.5-coder-14b-instruct-q4_k_m.gguf",
        "size_gb": 8.37,
        "description": "Advanced 14B coding model - requires 8+ GB VRAM (not recommended for 6GB VRAM)",
        "recommended": False,
    },
    "qwen2.5-coder-7b-q4": {
        "name": "Qwen2.5-Coder-7B-Instruct (Q4_K_M)",
        "url": "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "filename": "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "size_gb": 4.68,
        "description": "Balanced 7B coding model, good fit for 6GB VRAM (RECOMMENDED)",
        "recommended": True,
    },
    "qwen2.5-coder-7b-q5": {
        "name": "Qwen2.5-Coder-7B-Instruct (Q5_K_M)",
        "url": "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q5_k_m.gguf",
        "filename": "qwen2.5-coder-7b-instruct-q5_k_m.gguf",
        "size_gb": 5.53,
        "description": "Higher quality 7B, uses more VRAM",
        "recommended": False,
    },
    "phi-3.5-mini-q5": {
        "name": "Phi-3.5-mini-instruct (Q5_K_M)",
        "url": "https://huggingface.co/microsoft/Phi-3.5-mini-instruct-GGUF/resolve/main/phi-3.5-mini-instruct-q5_k_m.gguf",
        "filename": "phi-3.5-mini-instruct-q5_k_m.gguf",
        "size_gb": 3.12,
        "description": "Lightweight alternative, 3.8B parameters",
        "recommended": False,
    },
    "deepseek-coder-6.7b-q4": {
        "name": "DeepSeek-Coder-6.7B-Instruct (Q4_K_M)",
        "url": "https://huggingface.co/deepseek-ai/DeepSeek-Coder-6.7B-Instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct-q4_k_m.gguf",
        "filename": "deepseek-coder-6.7b-instruct-q4_k_m.gguf",
        "size_gb": 4.02,
        "description": "Excellent coding model from DeepSeek, good balance",
        "recommended": False,
    },
}


class DownloadProgressBar:
    """Progress bar for downloads."""
    
    def __init__(self):
        self.pbar = None
    
    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")
        self.pbar.update(block_size)


def download_model(model_key: str, model_dir: Path) -> bool:
    """Download a model file.
    
    Args:
        model_key: Key from MODELS dictionary.
        model_dir: Directory to save the model.
        
    Returns:
        True if download successful, False otherwise.
    """
    if model_key not in MODELS:
        print(f"Error: Unknown model '{model_key}'")
        print(f"Available models: {', '.join(MODELS.keys())}")
        return False
    
    model_info = MODELS[model_key]
    url = model_info["url"]
    filename = model_info["filename"]
    target_path = model_dir / filename
    
    print(f"\nDownloading: {model_info['name']}")
    print(f"Size: {model_info['size_gb']:.2f} GB")
    print(f"Target: {target_path}")
    
    # Create model directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    if target_path.exists():
        existing_size = target_path.stat().st_size / (1024**3)
        print(f"\nModel already exists ({existing_size:.2f} GB)")
        response = input("Re-download? (y/N): ").strip().lower()
        if response != 'y':
            print("Keeping existing file.")
            return True
    
    try:
        print(f"\nDownloading from: {url}")
        urlretrieve(url, str(target_path), DownloadProgressBar())
        print(f"\nDownload complete: {target_path}")
        return True
        
    except Exception as e:
        print(f"\nDownload failed: {e}")
        # Clean up partial download
        if target_path.exists():
            target_path.unlink()
        return False


def list_models():
    """List available models."""
    print("\nAvailable Models:")
    print("=" * 60)
    
    for key, info in MODELS.items():
        recommended = " (RECOMMENDED)" if info["recommended"] else ""
        print(f"\n{key}{recommended}")
        print(f"  Name: {info['name']}")
        print(f"  Size: {info['size_gb']:.2f} GB")
        print(f"  Description: {info['description']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download models for the local agent")
    parser.add_argument(
        "model",
        nargs="?",
        choices=list(MODELS.keys()),
        help="Model key to download"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available models"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        return
    
    if not args.model:
        list_models()
        print(f"\nUsage: python {__file__} <model_key>")
        print(f"Example: python {__file__} qwen2.5-coder-14b-q4")
        return
    
    model_dir = PROJECT_ROOT / "models"
    success = download_model(args.model, model_dir)
    
    if success:
        print("\n[DONE] Model ready for use.")
    else:
        print("\n[FAILED] Model download failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
