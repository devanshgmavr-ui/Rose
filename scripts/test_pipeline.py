#!/usr/bin/env python3
"""Full pipeline test for Stage 1.1."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.core.config import Config
from agent.core.agent import Agent


def main():
    config = Config()
    print(f"Config: {config.project_name} v{config.version} (Stage {config.stage})")
    print(f"Model: {config.model_name} @ {config.model_path}")
    print(f"GPU layers: {config.llm_gpu_layers}")

    agent = Agent(config)
    if agent.initialize():
        print("Agent initialized successfully!")
        response = agent.chat("Say hello in exactly 5 words.")
        print(f"Prompt: Say hello in exactly 5 words.")
        print(f"Response: {response.text.strip()}")
        print(f"Tokens: {response.tokens_used}")
        health = agent.health_check()
        initialized = health["agent"]["initialized"]
        print(f"Health: initialized={initialized}")
        agent.shutdown()
        print("Agent shutdown complete.")
        print()
        print("FULL PIPELINE: PASS")
    else:
        print("FAILED to initialize agent")
        print("FULL PIPELINE: FAIL")


if __name__ == "__main__":
    main()
