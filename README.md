<div align="center">

#  Rose

### Fully Local Autonomous AI Agent

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-2005%20passing-brightgreen.svg)](#testing)
[![Platform](https://img.shields.io/badge/Platform-Windows-0078d4.svg)](https://www.microsoft.com/windows)
[![GPU](https://img.shields.io/badge/GPU-CUDA-76b900.svg)](https://developer.nvidia.com/cuda-zone)

**100% local. Zero cloud. Full privacy. Your data never leaves your machine.**

[Installation](#installation) · [Features](#features) · [Architecture](#architecture) · [Roadmap](#roadmap) · [Contributing](#contributing)

</div>

---

## What is Rose?

Rose is an autonomous AI agent that runs entirely on your Windows PC. It uses a local Vision-Language model (Qwen2.5-VL-7B-Instruct) with NVIDIA GPU acceleration to understand natural language, analyze screenshots, and execute tasks through a comprehensive tool system — all without sending a single byte to the cloud.

```
You: "Take a screenshot of my desktop and tell me what applications are running"
Rose: *captures screen, analyzes it, identifies running apps, responds*
```

## Quick Start

### One-Click Install (Windows)

```batch
git clone https://github.com/devanshgmavr-ui/Rose.git
cd Rose
install.bat
```

The installer automatically:
1. Detects Python version and architecture
2. Detects NVIDIA GPU and CUDA version
3. Selects appropriate prebuilt wheels (no compiler needed)
4. Creates virtual environment
5. Downloads Qwen2.5-VL model files (~6 GB)
6. Configures Rose for your system
7. Verifies installation with health check

### Installer Options

```batch
install.bat                    # Full installation
install.bat --skip-models      # Skip model download (if already present)
install.bat --cpu-only         # Force CPU-only mode
install.bat --check-only       # Verify installation status
install.bat --quiet            # Minimal output
```

### Manual Install

```powershell
# Clone
git clone https://github.com/devanshgmavr-ui/Rose.git
cd Rose

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate

# Install with prebuilt wheels (recommended)
pip install -r requirements-runtime.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Or for CUDA 12.x:
# pip install -r requirements-runtime.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Download models
python scripts/download_models.py

# Run
python run.py
```

### Download Release

Grab the latest release from [Releases](https://github.com/devanshgmavr-ui/Rose/releases) — includes pre-built packages for Windows.

| Package | Description |
|---------|-------------|
| `rose-full-v1.0.0.zip` | Complete installation with all dependencies |
| `rose-core-v1.0.0.zip` | Core agent only (LLM + tools) |
| `rose-portable-v1.0.0.zip` | Portable version (no install needed) |

---

## Features

### 🧠 Intelligence
- **Local LLM Inference** — Qwen2.5-VL-7B-Instruct with CUDA GPU acceleration
- **Vision-Language Model** — Understands images and text together natively
- **Natural Language Planning** — Understands goals and breaks them into steps
- **Automatic Tool Selection** — Picks the right tool for each task
- **Multi-Step Execution** — Plans, executes, verifies, and adapts

### 🖥️ OS Automation
- **Screen Capture** — Capture and analyze your desktop
- **Mouse & Keyboard** — Controlled automation with safety limits
- **Window Management** — Enumerate, activate, minimize, move, resize
- **App Launching** — Start applications with arguments

### 🌐 Browser Control
- **Playwright Integration** — Headless browser automation
- **Page Navigation** — Open, read, and interact with web pages
- **Element Interaction** — Click, fill forms, select options
- **Screenshots** — Capture viewport, full-page, or specific elements

### 👁️ Vision
- **Image Analysis** — Understand what's on screen
- **Visual Grounding** — Translate vision into clickable coordinates
- **Observe/Act/Verify** — Safe loop for visual automation

### 🔍 OCR Pipeline
- **Modular OCR Provider** — Pluggable OCR abstraction with local Tesseract backend
- **Structured Results** — Text, confidence scores, and bounding boxes per word
- **Image Preprocessing** — RGB normalization, resizing, grayscale, contrast enhancement
- **Resource Limits** — Max image size, text chars, blocks, and timeout controls
- **Security Boundaries** — OCR output treated as untrusted, never executes commands
- **Grounding Integration** — OCR text with bounding boxes feeds into visual grounding

### 🧩 Multimodal Messaging
- **Content Types** — TextContent, ImageContent, OCRContent, GroundingContent, VisionSummaryContent
- **VisionContextBuilder** — Converts vision results to LLM-ready text context
- **Screenshot-to-Action Pipeline** — Screenshot → OCR → Grounding → LLM → Action
- **Autonomous Vision** — Vision context injected into autonomous task execution
- **Security** — All visual content wrapped in [UNTRUSTED] markers

### 📁 File & Code
- **File Automation** — Read, write, copy, move, search files
- **Code Execution** — Run Python in a sandboxed subprocess
- **CLI Control** — Execute shell commands (disabled by default)

### 🔒 Security
- **Permission System** — Every tool has configurable access levels
- **Workspace Boundary** — File access restricted to workspace
- **Confirmation Gates** — Dangerous actions require approval
- **Audit Logging** — All operations are logged
- **No Cloud** — Everything stays on your machine

---

## Architecture

```
                    USER
                      ↓
                 Agent API
                      ↓
             Context + Memory
                      ↓
             Task Understanding
                      ↓
              QWEN2.5-VL
             ↙     ↓      ↘
          TEXT   IMAGE   STATE
             ↘     ↓      ↙
                REASONING
                    ↓
                  PLAN
                    ↓
              TOOL SELECTION
                    ↓
             PERMISSION CHECK
                    ↓
                TOOL ROUTER
                    ↓
        ┌───────────┼────────────┐
        ↓           ↓            ↓
      OS Tools   Browser       Vision
        ↓           ↓            ↓
        └───────────┼────────────┘
                    ↓
                OBSERVATION
                    ↓
              QWEN2.5-VL
                    ↓
               VERIFICATION
                    ↓
              FAILURE RECOVERY
                    ↓
                 MEMORY
                    ↓
             FINAL RESPONSE
```

### Component Stack

| Layer | Component | Description |
|-------|-----------|-------------|
| **UI** | `agent/ui/` | PySide6 desktop app with chat, tasks, settings |
| **Web** | `agent/web/` | REST API, SSE streaming, ApplicationService |
| **Core** | `agent/core/` | Agent, Config, Health, Performance, Resources |
| **LLM** | `agent/llm/` | Local provider with CUDA, VL support via Llava16ChatHandler, VisionCapability abstraction |
| **Vision Pipeline** | `agent/media/vision_pipeline.py` | Unified VL + classical fallback routing |
| **Screen Understanding** | `agent/media/screen_understanding.py` | VL-based screen analysis |
| **System Prompt** | `agent/core/system_prompt.py` | Prompt injection defense, VL-aware prompts |
| **Memory** | `agent/memory/` | Session, conversation, long-term SQLite |
| **Tools** | `agent/tools/` | 15+ tools with permissions and audit |
| **Orchestration** | `agent/orchestration/` | Planner, executor, verifier, autonomous |
| **Media** | `agent/media/` | Vision, grounding, observe/act/verify |
| **OS Control** | `agent/os_control/` | Screen, mouse, keyboard, window |
| **Browser** | `agent/browser/` | Playwright sessions, navigation, interaction |

---

## Roadmap & Status

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Foundation (LLM, Memory, Tools, Orchestration) | ✅ Complete |
| **Phase 2** | OS Automation (Screen, Mouse, Keyboard, Window, Browser) | ✅ Complete |
| **Phase 3** | Perception (Vision, Grounding, Observe/Act/Verify) | ✅ Complete |
| **Phase 4** | Agent Intelligence (Planning, Tool Selection, Autonomous) | ✅ Complete |
| **Phase 5** | Application Service (Chat API, Tasks, Health, Events) | ✅ Complete |
| **Phase 6** | User Interface (PySide6 Desktop App) | ✅ Complete |
| **Phase 7** | End-to-End (run.py, First-Run, Startup) | ✅ Complete |
| **Phase 8** | Packaging (pyproject.toml, Installer, Releases) | ✅ Complete |
| **Phase 9** | Autonomous Tool Selection & Prompt Execution | ✅ Complete |
| **Phase 10** | Vision / Visual Understanding (RealVisionProvider) | ✅ Complete |
| **Phase 11** | Multimodal Agent Integration (Tool Selection) | ✅ Complete |
| **Phase 12** | Backend API & Event System | ✅ Complete |
| **Phase 13** | Observation, Verification & Failure Recovery | ✅ Complete |
| **Phase 14** | Memory Integration & Event Streaming Foundation | ✅ Complete |
| **Phase 15** | Vision + OCR Pipeline (OCRProvider, structured results) | ✅ Complete |
| **Phase 16** | Multimodal Messaging (VisionContextBuilder, screenshot-to-action) | ✅ Complete |

### Detailed Breakdown

<details>
<summary><strong>Phase 1: Foundation</strong></summary>

- 1.1 Local LLM Runtime — llama-cpp-python with CUDA
- 1.2 Memory & Context — Short-term + long-term SQLite
- 1.3 Tool Integration — Registry, router, permissions, sandbox
- 1.4 Task Orchestration — Plan → Execute → Verify loop
- 1.5 Media Architecture — Provider abstraction for vision/image/video
</details>

<details>
<summary><strong>Phase 2: OS Automation</strong></summary>

- 2.1 Screen Capture & System Info
- 2.2 Mouse & Keyboard Control
- 2.3 Window Management (enumerate, activate, minimize, move, resize, close)
- 2.4 Browser Automation (Playwright sessions, navigation, reading, interaction, screenshots)
</details>

<details>
<summary><strong>Phase 3: Perception</strong></summary>

- 3.1 Vision Analysis — Provider-agnostic image understanding
- 3.2 Visual Grounding — Translate vision to coordinates
- 3.3 Observe/Act/Verify — Safe visual automation loop
</details>

<details>
<summary><strong>Phase 4: Agent Intelligence</strong></summary>

- 4.1 Natural Language Tool Planning
- 4.2 Automatic Tool Selection (18 intent categories)
- 4.3 Multi-Step Autonomous Task Execution
</details>

<details>
<summary><strong>Phase 5-8: Application Layer</strong></summary>

- 5.1 ApplicationService — Single API between UI and backend
- 5.2 EventBus + SSE — Real-time event streaming
- 6.1 RoseUI — PySide6 desktop interface
- 7.1 run.py — CLI, web, and headless modes
- 7.2 First-run detection and setup
- 8.1 pyproject.toml, install.bat, release packages
</details>

<details>
<summary><strong>Phase 9: Autonomous Tool Selection & Prompt Execution</strong></summary>

- 9.1 CapabilityAnalyzer — Natural language → capability detection (15 capability types)
- 9.2 ToolScorer — Weighted scoring (capability, permissions, reliability, context, risk)
- 9.3 AutonomousTaskState — 11-state execution pipeline with step tracking
- 9.4 AutonomousLoop — observe→plan→execute→verify→replan cycle
- 9.5 Constraint Parsing — Prohibited tools, allowed files, minimize confirmations
- 9.6 ExecutionTrace — Concise UI-visible operational trace
- 9.7 67ms Controlled Typing — Enforced at keyboard execution layer
</details>

<details>
<summary><strong>Phase 10-14: Backend Hardening & Integration</strong></summary>

- 10.1 RealVisionProvider — Image preprocessing, color analysis, region detection, OCR integration
- 10.2 ImagePreprocessor — Safe loading, metadata extraction, format detection
- 11.1 Multimodal Integration — Vision tools wired into agent and autonomous tool selection
- 11.2 Config Vision Provider — Selectable "local" or "real" vision provider
- 11.3 Tool Catalog Enhancement — Refined keywords for vision, grounding, OCR
- 12.1 Backend API — Fixed path param extraction, new capabilities/permissions/system endpoints
- 12.2 ApplicationService — get_capabilities(), get_permissions(), get_system_status()
- 13.1 ObservationSystem — ACTION→OBSERVATION→STATE→VERIFICATION pipeline
- 13.2 FailureRecovery — Error classification and recovery decision-making (8 error categories)
- 13.3 Tool Fallbacks — Alternative tool selection on failure
- 14.1 MemoryIntegration — Task result storage, tool history, memory-augmented context
- 14.2 Memory Statistics — Tool and task execution stats
</details>

<details>
<summary><strong>Phase 15-16: Vision Pipeline & Multimodal Messaging</strong></summary>

- 15.1 OCRProvider Abstraction — Pluggable OCR engine interface
- 15.2 LocalOCRProvider — Tesseract-based OCR with language/config support
- 15.3 OCRResult/OCRBlock — Structured results with text, confidence, bounding boxes
- 15.4 OCR Security — Untrusted content markers, resource limits, no execution paths
- 15.5 VisionProvider Integration — RealVisionProvider uses OCRProvider abstraction
- 15.6 Grounding Integration — OCR text classified and grounded to screen coordinates
- 16.1 MultimodalMessage — TextContent, ImageContent, OCRContent, GroundingContent
- 16.2 VisionContextBuilder — Converts vision results to LLM-ready text context
- 16.3 Screenshot-to-Action Pipeline — Screenshot → OCR → Grounding → LLM → Action
- 16.4 Autonomous Vision — Vision context injected into autonomous task execution
</details>

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10 | Windows 11 |
| **CPU** | 4 cores | 8+ cores (Ryzen 7 / i7) |
| **RAM** | 8 GB | 16 GB |
| **GPU** | NVIDIA 4GB VRAM | RTX 4050+ (6GB+) |
| **Storage** | 10 GB free | 20 GB SSD |

### Development Config
- AMD Ryzen 7 / NVIDIA RTX 4050 (6GB)
- 16 GB RAM / Windows 11
- CUDA Toolkit 13.3

---

## Model

**Qwen2.5-VL-7B-Instruct** (Vision-Language Model)

| Property | Value |
|----------|-------|
| Parameters | 7 Billion |
| Quantization | Q4_K_M (main) + F16 (vision projector) |
| File Size | ~4.4 GB + ~1.3 GB mmproj |
| Context Length | 4096 tokens |
| GPU Layers | 28 (full offload) |
| Vision | Native image understanding via Llava16ChatHandler |
| License | Apache 2.0 |

### Setup

1. Download from [Hugging Face](https://huggingface.co/bartowski/Qwen_Qwen2.5-VL-7B-Instruct-GGUF)
2. Place both files in `models/` directory:
   - `Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` (main model)
   - `mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf` (vision projector)
3. Models auto-discovered on startup

```
Rose/
  models/
    Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf
    mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf
```

> **Future Goal:** Develop a custom trained model specifically for autonomous agent tasks.

---

## Configuration

### Environment Variables

```env
# LLM Settings
MODEL_PATH=./models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf
MMPROJ_PATH=./models/Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf
LLM_GPU_LAYERS=28
LLM_CONTEXT_LENGTH=4096

# OS Control
OS_CONTROL_ENABLED=true
SCREEN_CAPTURE_ENABLED=true
MOUSE_CONTROL_ENABLED=false
KEYBOARD_CONTROL_ENABLED=false
WINDOW_CONTROL_ENABLED=false

# Browser (disabled by default)
BROWSER_AUTOMATION_ENABLED=false
BROWSER_HEADLESS=true

# Vision (disabled by default)
VISION_ENABLED=false
VISION_PROVIDER=local

# OCR (enabled by default when vision is enabled)
OCR_ENABLED=true
OCR_PROVIDER=local_tesseract
OCR_LANGUAGE=eng
OCR_MAX_IMAGE_SIZE_MB=20
OCR_MAX_TEXT_CHARS=100000
OCR_MAX_BLOCKS=1000
OCR_TIMEOUT_MS=30000

# Multimodal (disabled by default)
MULTIMODAL_ENABLED=false
MULTIMODAL_MAX_OCR_CHARS=2000
MULTIMODAL_MAX_TARGETS=15
AUTONOMOUS_VISION_ENABLED=false
AUTONOMOUS_MAX_RETRIES=3
```

See `.env.example` for all options.

### Security Defaults

| Feature | Default | Why |
|---------|---------|-----|
| Mouse Control | Off | Prevents accidental clicks |
| Keyboard Control | Off | Prevents unintended typing |
| CLI Execution | Blocked | Prevents arbitrary commands |
| Browser | Off | Isolated from user profiles |
| Vision | Off | Requires explicit opt-in |
| Window Mutations | Confirmation | Requires approval |

---

## Deployment

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10 | Windows 11 |
| **Python** | 3.10 | 3.10-3.12 |
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8 GB | 16 GB |
| **GPU** | None (CPU mode) | NVIDIA 4GB+ VRAM |
| **Storage** | 5 GB free | 10 GB SSD |

### Installation Strategies

Rose uses prebuilt wheels to avoid requiring compilation tools:

| Mode | Wheel Index | Requirements |
|------|-------------|--------------|
| **CPU** | `whl/cpu` | Python 3.10-3.12 |
| **CUDA 11.8** | `whl/cu118` | NVIDIA GPU, CUDA 11.8 |
| **CUDA 12.1** | `whl/cu121` | NVIDIA GPU, CUDA 12.1+ |
| **CUDA 12.2** | `whl/cu122` | NVIDIA GPU, CUDA 12.2+ |
| **CUDA 12.3** | `whl/cu123` | NVIDIA GPU, CUDA 12.3+ |
| **CUDA 12.4** | `whl/cu124` | NVIDIA GPU, CUDA 12.4+ |
| **CUDA 12.5** | `whl/cu125` | NVIDIA GPU, CUDA 12.5+ |
| **CUDA 13.0** | `whl/cu130` | NVIDIA GPU, CUDA 13.0+ |
| **CUDA 13.2** | `whl/cu132` | NVIDIA GPU, CUDA 13.2+ |

### Supported Python Versions

| Python | Prebuilt Wheels | Notes |
|--------|-----------------|-------|
| 3.10 | Yes | Recommended |
| 3.11 | Yes | Recommended |
| 3.12 | Yes | Recommended |
| 3.13 | Limited | May require compilation |
| 3.14+ | No | Not supported |

### Troubleshooting Installation

**Problem: `CMAKE_C_COMPILER not set` or `nmake not found`**

This happens when the installer tries to compile llama-cpp-python from source because no prebuilt wheel is available.

Solutions:
1. Use Python 3.10, 3.11, or 3.12 (recommended)
2. Use `--cpu-only` flag if you don't need GPU acceleration
3. Install Visual Studio Build Tools for source compilation
4. Check that your CUDA version is supported (11.8, 12.1-12.5, 13.0, 13.2)

**Problem: `Could not find a version that satisfies requirement`**

This means no prebuilt wheel exists for your Python + CUDA combination.

Solutions:
1. Use Python 3.10, 3.11, or 3.12
2. Check your CUDA version with `nvidia-smi`
3. Use `--cpu-only` flag

**Problem: `No module named 'playwright'`**

Browser automation is optional and not installed by default.

Solution: `pip install playwright && python -m playwright install chromium`

**Problem: Model download fails**

Solutions:
1. Check internet connection
2. Try `python scripts/download_models.py` directly
3. Manually download from HuggingFace and place in `models/` directory

**Problem: `CUDA error: out of memory`**

Your GPU doesn't have enough VRAM for the model.

Solutions:
1. Close other GPU-intensive applications
2. Reduce `LLM_GPU_LAYERS` in `.env`
3. Use a smaller quantization (Q3_K_S or IQ3_M)

---

## Testing

**2005 unit tests** — all passing

```powershell
# Run all tests
python -m pytest tests/unit/ -v

# Run backend smoke test (18 checks)
python scripts/rose_smoke_test.py

# Run end-to-end tests (12 scenarios)
python scripts/rose_e2e_tests.py

# Run performance validation
python scripts/rose_perf_test.py

# Run VL pipeline tests
python -m pytest tests/unit/test_vl_pipeline.py -v

# Run multimodal message tests
python -m pytest tests/unit/test_multimodal.py -v

# Run vision→LLM integration tests
python -m pytest tests/unit/test_vision_llm_integration.py -v

# Run OCR tests specifically
python -m pytest tests/unit/test_ocr.py -v

# Run Vision + OCR integration tests
python -m pytest tests/unit/test_vision_ocr_integration.py -v

# Run real OCR test (requires Tesseract installed)
python -m pytest tests/unit/test_real_ocr.py -v

# Run Phase 9 tests specifically
python -m pytest tests/unit/test_autonomous_selection.py -v

# Run specific module
python -m pytest tests/unit/test_vision.py -v

# Run with coverage
python -m pytest tests/unit/ --cov=agent
```

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| Core (Agent, Config) | 45+ | Full |
| LLM Provider | 30+ | Full |
| Memory System | 40+ | Full |
| Tool System | 80+ | Full |
| Orchestration | 130+ | Full |
| Media/Vision | 85+ | Full |
| OS Control | 100+ | Full |
| Browser | 120+ | Full |
| Web/API | 50+ | Full |
| Application Service | 63 | Full |
| UI Components | 83 | Full |
| Startup/Packaging | 67 | Full |
| Phase 9 Autonomous | 101 | Full |
| Phase 10 Vision | 31 | Full |
| Phase 11 Multimodal | 32 | Full |
| Phase 12 Backend API | 58 | Full |
| Phase 13 Observ/Recovery | 55 | Full |
| Phase 14 Memory Integration | 24 | Full |
| OCR Pipeline | 62 | Full |
| Multimodal Messages | 55 | Full |
| Vision→LLM Integration | 6 | Full |
| VL Pipeline (Phase 17) | 76 | Full |
| Production Hardening | 17 | Full |
| **Total** | **2005** | **Full** |

---

## Security Model

### Permission Levels

```
ALLOW                    → No confirmation needed
REQUIRE_CONFIRMATION     → User must approve
DENIED                   → Blocked entirely
```

### Safety Features

- **Workspace boundary** — File access restricted to `workspace/`
- **Sandbox execution** — Python runs in subprocess
- **Coordinate validation** — Mouse/keyboard checked against screen bounds
- **Restricted shortcuts** — Ctrl+Alt+Del, Alt+F4, etc. blocked
- **HWND validation** — Window handles verified before mutations
- **Protected windows** — System-critical windows cannot be modified
- **Browser isolation** — No access to user Chrome/Edge profiles
- **URL sanitization** — Sensitive parameters redacted in logs
- **No binary in logs** — Image data never stored in audit trail

---

## Repository Structure

```
Rose/
├── agent/                    # Core agent code
│   ├── core/                 # Agent, Config, Health, Performance
│   ├── llm/                  # LLM provider, model optimizer
│   ├── memory/               # Session, conversation, long-term
│   ├── tools/                # Tool registry, router, permissions
│   ├── orchestration/        # Planner, executor, autonomous
│   ├── media/                # Vision, grounding, OAV loop
│   ├── os_control/           # Screen, mouse, keyboard, window
│   ├── browser/              # Playwright browser automation
│   ├── web/                  # REST API, ApplicationService
│   └── ui/                   # PySide6 desktop interface
├── tests/                    # Test suites
│   ├── unit/                 # 1459 unit tests
│   └── integration/          # Integration tests
├── run.py                    # Main entry point
├── install.bat               # Windows installer
├── pyproject.toml            # Package configuration
├── requirements-runtime.txt  # Runtime dependencies
├── .env.example              # Config template
└── README.md                 # This file
```

---

## Usage

### Interactive CLI
```powershell
python run.py
# Type messages, Rose responds
# Type 'help' for commands, 'quit' to exit
```

### Web Interface
```powershell
python run.py --web
# Open http://127.0.0.1:8080 in your browser
```

### Headless Mode
```powershell
python run.py --headless
# For testing and automation
```

### Check Status
```powershell
python run.py status
# Shows system health, dependencies, model info
```

---

## Releases

Download pre-built packages from [GitHub Releases](https://github.com/devanshgmavr-ui/Rose/releases):

| Release | Date | Description |
|---------|------|-------------|
| [v1.0.0](https://github.com/devanshgmavr-ui/Rose/releases/tag/v1.0.0) | Aug 2025 | Initial release — all 8 phases complete |

### Package Contents

**`rose-full-v1.0.0.zip`** — Complete installation
```
Rose/
├── agent/              # All agent modules
├── tests/              # Test suites
├── run.py              # Entry point
├── install.bat         # Installer
├── pyproject.toml      # Package config
├── requirements-runtime.txt  # Dependencies
├── .env.example        # Config template
└── README.md           # Documentation
```

**`rose-core-v1.0.0.zip`** — Core only (smaller download)
```
Rose/
├── agent/core/         # Agent core
├── agent/llm/          # LLM provider
├── agent/memory/       # Memory system
├── agent/tools/        # Tool system
├── run.py
└── requirements-runtime.txt
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Run tests (`python -m pytest tests/unit/ -v`)
4. Submit a pull request

---

<div align="center">

**Built with local AI. No cloud required.**

[⭐ Star this repo](https://github.com/devanshgmavr-ui/Rose/stargazers) · [🐛 Report Bug](https://github.com/devanshgmavr-ui/Rose/issues) · [📦 Releases](https://github.com/devanshgmavr-ui/Rose/releases)

</div>
