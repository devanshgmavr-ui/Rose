# Rose

Local Autonomous AI Agent

## Overview

Rose is a fully local autonomous AI agent built for Windows with GPU acceleration. It runs entirely on your machine without requiring cloud services, keeping all data private and local.

The project is being developed incrementally through staged milestones, with each stage adding new capabilities while maintaining a stable, tested foundation.

## Project Goals

**Current Capabilities (Completed):**
- Local LLM inference with GPU acceleration
- Long-term memory and context management
- Tool system with permissions and sandboxing
- Task planning and orchestration
- Multimodal media architecture
- Screen capture and system information
- Mouse and keyboard control
- Window management (enumerate, activate, minimize, restore, maximize, close, move, resize)
- Browser foundation (Playwright session management, isolated contexts)
- Browser page reading (extract text content from pages with security wrapping)
- Browser interaction (inspect, click, fill, select, press, wait with controlled targeting)
- Browser screenshots (viewport, full-page, element capture with secure storage)

**Planned Capabilities (In Development):**
- Browser automation
- Image understanding and vision analysis
- Image generation
- Video generation
- Voice input/output
- Backend API and web UI
- Cloud scaling options

**Stage 3.1 Vision Analysis Capabilities:**
- Provider-agnostic vision analysis architecture
- Structured output with detected elements, bounding boxes, confidence levels
- Workspace-boundary image validation
- Image format, size, and dimension validation
- Untrusted content markers for security
- Vision permission system with configurable confirmation
- Health checking and statistics
- Text description generation with security wrapping

**Stage 3.2 Visual Grounding Capabilities:**
- VisualGrounder translates vision results into actionable coordinates
- GroundedTarget with center point, bounding box, target type, confidence
- Target classification (button, link, text_field, icon, menu, etc.)
- Coordinate clamping to screen bounds
- Target validation (bounds, confidence, staleness)
- Ground/validate actions with untrusted data markers

**Stage 3.3 Observe/Act/Verify Capabilities:**
- Safe loop for visual automation with configurable limits
- Maximum iteration count (prevents infinite loops)
- Maximum action count
- Timeout enforcement
- Goal detection and achievement tracking
- Cancel support
- Step-by-step execution logging

## Architecture

```
User
  |
  v
Agent
  |
  v
LLM (Qwen 7B - Local GPU)
  |
  v
Planner
  |
  v
Tool Router
  |
  +---> Permissions
  |
  +---> Audit Logger
  |
  v
Tools
  |
  +---> Filesystem (workspace only)
  +---> Python Sandbox (subprocess)
  +---> CLI (disabled by default)
  +---> Screen Capture
  +---> System Info
  +---> Mouse Control
  +---> Keyboard Control
  +---> Window Management
  +---> Browser Session Management
  +---> Media (vision/image/video)
  |
  v
Windows / Filesystem / Browser / Sandbox
```

### Components

| Layer | Description |
|-------|-------------|
| **LLM** | Local inference via llama-cpp-python with CUDA GPU acceleration |
| **Memory** | Session management, conversation history, long-term SQLite storage |
| **Tools** | Permission-based tool system with audit logging |
| **Orchestration** | Task planning, execution, verification, and persistence |
| **Media** | Multimodal providers for vision, image, and video |
| **OS Control** | Screen capture, system info, mouse, keyboard, window management |
| **Browser** | Playwright-based browser session management (disabled by default) |

## Current Status

| Stage | Description | Status |
|-------|-------------|--------|
| 1.1 | Local LLM Runtime | COMPLETE |
| 1.2 | Memory & Context Management | COMPLETE |
| 1.3 | Tool Integration & Sandboxing | COMPLETE |
| 1.4 | Task Orchestration & Verification | COMPLETE |
| 1.5 | Media Architecture | COMPLETE |
| 2.1 | Screen Capture & System Information | COMPLETE |
| 2.2 | Mouse & Keyboard Control | COMPLETE |
| 2.3.1 | Window Architecture & Research | COMPLETE |
| 2.3.2 | Window Enumeration | COMPLETE |
| 2.3.3 | Window Control Foundation | COMPLETE |
| 2.3.4 | Advanced Window Operations | COMPLETE |
| 2.4.1 | Browser Foundation / Session Management | COMPLETE |
| 2.4.2 | Browser Navigation | COMPLETE |
| 2.4.3 | Browser Page Reading | COMPLETE |
| 2.4.4 | Browser Interaction | COMPLETE |
| 2.4.5 | Browser Screenshots | COMPLETE |
| 2.4 | Browser Automation | IN PROGRESS |
| 3.1 | Vision Analysis | COMPLETE |
| 3.2 | Visual Grounding | COMPLETE |
| 3.3 | Observe/Act/Verify | COMPLETE |
| 4.1 | Natural Language Tool Planning | PLANNED |
| 4.1 | Backend API | PLANNED |
| 4.2 | Web UI | PLANNED |
| 5.1 | Voice I/O | PLANNED |
| 6.1 | Cloud Scaling | PLANNED |

## Hardware Requirements

### Development Configuration

- **OS:** Windows 11
- **CPU:** AMD Ryzen 7
- **RAM:** 16 GB
- **GPU:** NVIDIA GeForce RTX 4050 (6 GB VRAM)
- **CUDA:** Enabled for local inference

### Minimum Requirements

- Windows 10 or later
- 8 GB RAM
- NVIDIA GPU with CUDA support (4 GB+ VRAM recommended)
- 10 GB free disk space

## Model

**Current Model:**
- Qwen2.5-Coder-7B-Instruct
- Q4_K_M quantization
- ~4.36 GB file size
- 4096 token context
- 28 GPU layers (full offload)

**Model Placement:**
```
Rose/
  models/
    qwen2.5-coder-7b-instruct-q4_k_m.gguf
```

> **Note:** The current implementation uses Qwen as the underlying local base model. The long-term project goal is to develop a customized/own model through future model development, training, and/or fine-tuning work.

## Installation

### 1. Clone Repository

```powershell
git clone https://github.com/YOUR_USERNAME/Rose.git
cd Rose
```

### 2. Create Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure Environment

```powershell
copy .env.example .env
# Edit .env with your settings
```

### 5. Place Model

Download the model file and place it in the `models/` directory:
```
models/qwen2.5-coder-7b-instruct-q4_k_m.gguf
```

### 6. Verify Installation

```powershell
python scripts/health_check.py
```

### 7. Run Tests

```powershell
python -m pytest tests/unit/ -v
```

## Configuration

### Environment Variables

Configuration is managed through `.env` file (not committed to Git).

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./models/...gguf` | Path to model file |
| `LLM_GPU_LAYERS` | `28` | GPU layers for offload |
| `MOUSE_CONTROL_ENABLED` | `false` | Enable mouse control |
| `KEYBOARD_CONTROL_ENABLED` | `false` | Enable keyboard control |

See `.env.example` for all available options.

### Security Notes

- `.env` contains local configuration only (no secrets)
- `.env` is git-ignored and never committed
- Model files are git-ignored (too large for Git)
- All runtime data stays local

## Security Model

### Permissions

| Permission | Default | Confirmation |
|------------|---------|--------------|
| `filesystem.read` | ALLOW | No |
| `filesystem.write` | REQUIRE_CONFIRMATION | Yes |
| `code.execute` | REQUIRE_CONFIRMATION | Yes |
| `command.execute` | DENIED | Blocked |
| `os.screen_capture` | ALLOW | No |
| `os.system_info` | ALLOW | No |
| `os.mouse` | REQUIRE_CONFIRMATION | Yes |
| `os.keyboard` | REQUIRE_CONFIRMATION | Yes |
| `os.window` | ALLOW | No (mutations require confirmation) |
| `browser.session` | DENIED | Yes (requires browser enabled) |
| `browser.navigation` | DENIED | Yes (requires browser enabled) |
| `browser.screenshot` | DENIED | Yes (requires browser + screenshot enabled) |
| `vision.analyze` | DENIED | Yes (requires vision enabled) |

### Safety Features

- **Workspace boundary:** Filesystem access restricted to `workspace/`
- **Sandbox execution:** Python code runs in subprocess (not kernel-isolated)
- **CLI disabled:** Command execution blocked by default
- **Mouse/keyboard disabled:** Must be explicitly enabled
- **Window control disabled:** Must be explicitly enabled
- **Window close disabled:** Must be explicitly enabled
- **Window move disabled:** Must be explicitly enabled
- **Window resize disabled:** Must be explicitly enabled
- **Coordinate validation:** Actions validated against screen bounds
- **Restricted shortcuts:** Dangerous key combinations blocked
- **HWND validation:** Window handles validated before mutations
- **Protected windows:** System-critical windows cannot be modified
- **Confirmation required:** All window mutations require explicit confirmation
- **Browser disabled:** Browser automation disabled by default
- **Browser isolation:** Playwright contexts isolated from user profiles
- **Browser session limit:** Maximum concurrent sessions enforced
- **Browser page limit:** Maximum pages per session enforced
- **Browser navigation:** HTTP/HTTPS only, unsupported schemes blocked
- **Browser page reading:** Content treated as untrusted, wrapped with security markers
- **Browser interaction:** Controlled element targeting, value redaction, no JavaScript execution
- **Browser inspect:** Read-only element inspection with configurable limits
- **Browser screenshots:** Viewport, full-page, and element capture with dimension/size limits
- **Screenshot storage:** Secure workspace-boundary storage, PNG-only format
- **Screenshot limits:** Configurable width, height, file size, and per-request count limits
- **URL sanitization:** Sensitive query parameters redacted in logs
- **Vision disabled:** Vision analysis disabled by default
- **Vision workspace boundary:** Image paths validated against workspace
- **Vision content markers:** All visual content wrapped with untrusted markers
- **Vision permissions:** Dedicated vision.analyze permission with confirmation
- **Vision validation:** Image format, size, and dimension limits enforced
- **No binary in logs:** Image data never stored in audit logs
- **No browser profiles:** No access to user Chrome/Edge data
- **Action limits:** Configurable limits per request
- **Audit logging:** All tool calls logged

### Limitations

- Subprocess isolation is NOT kernel-level sandboxing
- OS automation is intentionally restricted
- Mouse/keyboard require explicit opt-in
- No arbitrary command execution

## Testing

**Current Result:** 923/923 tests passing (54 vision, 34 grounding, 27 OAV tests)

```powershell
# Run all unit tests
python -m pytest tests/unit/ -v

# Run specific test suite
python -m pytest tests/unit/test_os_control.py -v

# Run browser tests
python -m pytest tests/unit/test_browser.py -v
```

### Test Coverage

| Module | Tests |
|--------|-------|
| Configuration | Config loading, defaults, env override |
| LLM Interface | Provider initialization, generation |
| Memory | Session, conversation, long-term storage |
| Tools | Registry, router, permissions, audit |
| Orchestration | Planning, execution, verification |
| Media | Storage, routing, providers |
| Vision | Provider, analyzer, permissions, tool, validation |
| OS Control | Screen, system, mouse, keyboard, window |
| Browser | Session management, navigation, page reading, interaction, screenshots, URL validation, models, permissions, tools |

## Current Capabilities

The agent can:
- Run local LLM inference with GPU acceleration
- Maintain conversation memory across sessions
- Store and retrieve long-term memories
- Plan and execute multi-step tasks
- Capture screenshots of the desktop
- Get system information (OS, CPU, memory, screen)
- Move mouse and click (when enabled)
- Type text and press keys (when enabled)
- Enumerate and list windows
- Get active window information
- Activate (focus) a specific window
- Minimize, restore, and maximize windows
- Gracefully close windows (WM_CLOSE)
- Move windows to new positions
- Resize windows
- Set window bounds (position + size)
- Protected system window detection
- Create isolated browser sessions (Playwright)
- List active browser sessions
- Close browser sessions
- Navigate browser pages to HTTP/HTTPS URLs
- URL scheme validation (HTTP/HTTPS only)
- Sensitive URL parameter sanitization in logs
- Vision analysis with structured output
- Image validation (format, size, dimensions)
- Workspace-boundary image path validation
- Untrusted content markers for visual data
- Execute Python code in sandbox
- Read/write files in workspace
- Store and retrieve media files

## Current Limitations

- Video provider is a stub (requires local model)
- Image generation is a stub (requires local model)
- Vision analysis works with metadata only (no real vision model loaded)
- Python sandbox is subprocess-level, not kernel-isolated
- OS automation is intentionally restricted
- Mouse/keyboard disabled by default
- Window control disabled by default
- Window close/move/resize disabled by default
- Window mutations require explicit confirmation
- WM_CLOSE does not guarantee application termination
- Protected system windows cannot be modified
- No process management or termination
- Browser automation disabled by default
- Browser page reading is available (with security wrapping)
- Browser interaction is available (inspect, click, fill, select, press, wait)
- Browser screenshots are available (viewport, full-page, element)
- No JavaScript execution is supported through browser tools
- Browser interaction not yet implemented
- No persistent browser profiles
- 4K token context limit (VRAM constrained)
- Single monitor support only

## Roadmap

### Stage 1: Foundation (COMPLETE)
- 1.1 Local LLM Runtime
- 1.2 Memory & Context
- 1.3 Tools & Sandboxing
- 1.4 Task Orchestration
- 1.5 Media Architecture

### Stage 2: OS Automation (COMPLETE)
- 2.1 Screen/System Control (COMPLETE)
- 2.2 Mouse/Keyboard Control (COMPLETE)
- 2.3 Window Management (COMPLETE)
  - 2.3.1 Window Architecture (COMPLETE)
  - 2.3.2 Window Enumeration (COMPLETE)
  - 2.3.3 Window Control Foundation (COMPLETE)
  - 2.3.4 Advanced Window Operations (COMPLETE)
- 2.4 Browser Automation (COMPLETE)
  - 2.4.1 Browser Foundation / Session Management (COMPLETE)
  - 2.4.2 Browser Navigation (COMPLETE)
  - 2.4.3 Browser Page Reading (COMPLETE)
  - 2.4.4 Browser Interaction (COMPLETE)
  - 2.4.5 Browser Screenshots (COMPLETE)

### Stage 3: Perception (COMPLETE)
- 3.1 Vision Analysis (COMPLETE)
- 3.2 Visual Grounding (COMPLETE)
- 3.3 Observe/Act/Verify (COMPLETE)

### Stage 4: Agent Intelligence (IN PROGRESS)
- 4.1 Natural Language Tool Planning (PLANNED)
- 4.2 Automatic Tool Selection (PLANNED)
- 4.3 Multi-Step Autonomous Tasks (PLANNED)

### Stage 5: Voice (PLANNED)
- 5.1 Voice Input/Output

### Stage 6: Scale (PLANNED)
- 6.1 Cloud Scaling

## Repository Structure

```
Rose/
+-- agent/                    # Core agent logic
|   +-- core/                 # Configuration and agent
|   +-- llm/                  # LLM abstraction layer
|   +-- memory/               # Memory systems
|   +-- tools/                # Tool system
|   +-- orchestration/        # Task orchestration
|   +-- media/                # Multimodal media
|   +-- os_control/           # OS automation
|   +-- router/               # Input/output routing
+-- tests/                    # Test suites
|   +-- unit/                 # Unit tests
|   +-- integration/          # Integration tests
+-- scripts/                  # Utility scripts
+-- configs/                  # Configuration files
+-- models/                   # Model files (git-ignored)
+-- workspace/                # Agent workspace (git-ignored)
+-- .env.example              # Environment template
+-- .gitignore                # Git ignore rules
+-- requirements.txt          # Python dependencies
+-- README.md                 # This file
```

## License

License: To be determined.

## Contributing

This project is currently in active development. Contributions and feedback are welcome.
