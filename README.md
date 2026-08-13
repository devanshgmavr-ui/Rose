<div align="center">

#  Rose

### Fully Local Autonomous AI Agent

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-1528%20passing-brightgreen.svg)](#testing)
[![Platform](https://img.shields.io/badge/Platform-Windows-0078d4.svg)](https://www.microsoft.com/windows)
[![GPU](https://img.shields.io/badge/GPU-CUDA-76b900.svg)](https://developer.nvidia.com/cuda-zone)

**100% local. Zero cloud. Full privacy. Your data never leaves your machine.**

[Installation](#installation) · [Features](#features) · [Architecture](#architecture) · [Roadmap](#roadmap) · [Contributing](#contributing)

</div>

---

## What is Rose?

Rose is an autonomous AI agent that runs entirely on your Windows PC. It uses a local LLM (Qwen2.5-Coder-7B) with NVIDIA GPU acceleration to understand natural language, plan tasks, and execute them through a comprehensive tool system — all without sending a single byte to the cloud.

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

### Manual Install

```powershell
# Clone
git clone https://github.com/devanshgmavr-ui/Rose.git
cd Rose

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt

# Place your model
# models/qwen2.5-coder-7b-instruct-q4_k_m.gguf

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
- **Local LLM Inference** — Qwen2.5-Coder-7B with CUDA GPU acceleration
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
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│              (CLI / Web / PySide6 GUI)                   │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│              Application Service (Phase 5)               │
│     Chat API · Task API · Health API · Event Bus         │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                  Rose Agent Core                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │   LLM    │ │  Memory  │ │ Planner  │ │ Verifier │   │
│  │ Qwen 7B  │ │ Short +  │ │ NL →     │ │ Result   │   │
│  │ CUDA GPU │ │ Long Term│ │ Plan     │ │ Check    │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                  Tool Router                              │
│          Permissions · Audit · Confirmation               │
└─────────────────────┬───────────────────────────────────┘
                      │
    ┌─────────┬───────┼───────┬─────────┬─────────┐
    │         │       │       │         │         │
┌───▼──┐ ┌───▼──┐ ┌──▼──┐ ┌──▼──┐ ┌───▼──┐ ┌───▼──┐
│Files │ │Code │ │ CLI │ │ OS  │ │Browser│ │Vision│
│R/W/S │ │Exec │ │Exec │ │Ctrl │ │PW    │ │Analyze│
└──────┘ └─────┘ └─────┘ └─────┘ └──────┘ └──────┘
    │         │       │       │         │         │
┌───▼─────────▼───────▼───────▼─────────▼─────────▼───┐
│              Windows / Filesystem / Browser            │
└──────────────────────────────────────────────────────┘
```

### Component Stack

| Layer | Component | Description |
|-------|-----------|-------------|
| **UI** | `agent/ui/` | PySide6 desktop app with chat, tasks, settings |
| **Web** | `agent/web/` | REST API, SSE streaming, ApplicationService |
| **Core** | `agent/core/` | Agent, Config, Health, Performance, Resources |
| **LLM** | `agent/llm/` | Local provider with CUDA, model optimizer |
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

**Qwen2.5-Coder-7B-Instruct** (Q4_K_M quantization)

| Property | Value |
|----------|-------|
| Parameters | 7 Billion |
| Quantization | Q4_K_M |
| File Size | ~4.36 GB |
| Context Length | 4096 tokens |
| GPU Layers | 28 (full offload) |
| License | Apache 2.0 |

### Setup

1. Download from [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF)
2. Place in `models/` directory
3. The model auto-discovered on startup

```
Rose/
  models/
    qwen2.5-coder-7b-instruct-q4_k_m.gguf
```

> **Future Goal:** Develop a custom trained model specifically for autonomous agent tasks.

---

## Configuration

### Environment Variables

```env
# LLM Settings
MODEL_PATH=./models/qwen2.5-coder-7b-instruct-q4_k_m.gguf
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

## Testing

**1528 unit tests** — all passing

```powershell
# Run all tests
python -m pytest tests/unit/ -v

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
| **Total** | **1459** | **Full** |

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
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
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
├── requirements.txt    # Dependencies
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
└── requirements.txt
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
