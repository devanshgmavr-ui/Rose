#!/usr/bin/env python3
"""GPU inference test with VRAM monitoring for Stage 1.1."""

import sys
import os
import time
import ctypes
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add CUDA bin directories to DLL search path using Windows API
# This is needed for implicit DLL dependencies (ggml-cuda.dll -> cudart64_13.dll)
_cuda_bin_dirs = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin",
]
for _cuda_dir in _cuda_bin_dirs:
    if os.path.isdir(_cuda_dir):
        try:
            ctypes.windll.kernel32.SetDllDirectoryW(_cuda_dir)
        except Exception:
            pass
        # Also add to PATH for child processes
        os.environ["PATH"] = _cuda_dir + ";" + os.environ.get("PATH", "")


def get_vram_usage():
    """Get current VRAM usage via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 3:
                return {
                    "used_mb": int(float(parts[0])),
                    "total_mb": int(float(parts[1])),
                    "gpu_util": int(float(parts[2])),
                }
    except Exception:
        pass
    return None


def main():
    from llama_cpp import Llama
    from agent.core.config import Config

    config = Config()
    model_path = str(config.get_model_full_path())

    print("=" * 60)
    print("STAGE 1.1 GPU INFERENCE TEST")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Path: {model_path}")
    print(f"File size: {Path(model_path).stat().st_size / (1024**3):.2f} GB")
    print(f"Context: {config.model_context_length}")
    print(f"GPU layers: {config.llm_gpu_layers}")
    print()

    # Check VRAM before loading
    print("--- VRAM Before Model Load ---")
    vram = get_vram_usage()
    if vram:
        print(f"VRAM used: {vram['used_mb']} MB / {vram['total_mb']} MB")
        print(f"GPU util: {vram['gpu_util']}%")
    else:
        print("Could not read VRAM usage")
    print()

    # Load model with GPU offload
    print("Loading model with GPU offload...")
    start = time.time()
    llama = Llama(
        model_path=model_path,
        n_ctx=config.model_context_length,
        n_gpu_layers=config.llm_gpu_layers,
        n_batch=config.llm_batch_size,
        verbose=False,
        embedding=True,
    )
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")
    print()

    # Check VRAM after loading
    print("--- VRAM After Model Load ---")
    vram = get_vram_usage()
    if vram:
        print(f"VRAM used: {vram['used_mb']} MB / {vram['total_mb']} MB")
        print(f"GPU util: {vram['gpu_util']}%")
    else:
        print("Could not read VRAM usage")
    print()

    # Test 1: Basic generation
    print("--- Test 1: Basic Generation (GPU) ---")
    prompt = "What is Python? Answer in 2 sentences."
    start = time.time()
    response = llama(prompt=prompt, max_tokens=100, temperature=0.7, echo=False)
    elapsed = time.time() - start
    text = response["choices"][0]["text"]
    tokens = response.get("usage", {}).get("total_tokens", 0)
    print(f"Prompt: {prompt}")
    print(f"Response: {text.strip()}")
    print(f"Tokens: {tokens} | Time: {elapsed:.2f}s | Speed: {tokens/elapsed:.1f} tok/s")
    assert len(text.strip()) > 0, "Empty response"
    assert tokens > 0, "No tokens reported"
    print("PASS")
    print()

    # Check VRAM during/after inference
    print("--- VRAM After Inference ---")
    vram = get_vram_usage()
    if vram:
        print(f"VRAM used: {vram['used_mb']} MB / {vram['total_mb']} MB")
        print(f"GPU util: {vram['gpu_util']}%")
    print()

    # Test 2: Chat completion
    print("--- Test 2: Chat Completion (GPU) ---")
    messages = [{"role": "user", "content": "What is 2 + 2? Reply with just the number."}]
    start = time.time()
    response = llama.create_chat_completion(messages=messages, max_tokens=50, temperature=0.7)
    elapsed = time.time() - start
    text = response["choices"][0]["message"]["content"]
    tokens = response.get("usage", {}).get("total_tokens", 0)
    print(f"Prompt: {messages[0]['content']}")
    print(f"Response: {text.strip()}")
    print(f"Tokens: {tokens} | Time: {elapsed:.2f}s | Speed: {tokens/elapsed:.1f} tok/s")
    assert len(text.strip()) > 0, "Empty response"
    assert tokens > 0, "No tokens reported"
    print("PASS")
    print()

    # Test 3: Coding prompt
    print("--- Test 3: Coding Prompt (GPU) ---")
    messages = [
        {
            "role": "user",
            "content": "Write a Python function that checks if a number is prime. Just the function, no explanation.",
        }
    ]
    start = time.time()
    response = llama.create_chat_completion(messages=messages, max_tokens=200, temperature=0.3)
    elapsed = time.time() - start
    text = response["choices"][0]["message"]["content"]
    tokens = response.get("usage", {}).get("total_tokens", 0)
    print(f"Response:\n{text.strip()}")
    print(f"Tokens: {tokens} | Time: {elapsed:.2f}s | Speed: {tokens/elapsed:.1f} tok/s")
    assert len(text.strip()) > 0, "Empty response"
    assert tokens > 0, "No tokens reported"
    assert "def " in text, "No function definition in response"
    print("PASS")
    print()

    # Final VRAM check
    print("--- Final VRAM Usage ---")
    vram = get_vram_usage()
    if vram:
        print(f"VRAM used: {vram['used_mb']} MB / {vram['total_mb']} MB")
        print(f"GPU util: {vram['gpu_util']}%")
    print()

    del llama
    print("=" * 60)
    print("ALL GPU INFERENCE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
