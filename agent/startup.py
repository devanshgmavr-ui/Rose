"""Rose startup and first-run detection.

Phase 7 - Real End-to-End Application.

Handles:
- First-run detection and setup
- Dependency validation
- Model discovery and validation
- Configuration generation
- Environment setup
"""

import os
import sys
import json
import time
import logging
import platform
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StartupConfig:
    """Configuration for Rose startup."""
    workspace_dir: str = ""
    models_dir: str = ""
    data_dir: str = ""
    config_file: str = ""
    log_level: str = "INFO"
    headless: bool = False
    web_mode: bool = False
    web_port: int = 8080
    skip_model_check: bool = False
    skip_dependency_check: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace_dir": self.workspace_dir,
            "models_dir": self.models_dir,
            "data_dir": self.data_dir,
            "config_file": self.config_file,
            "log_level": self.log_level,
            "headless": self.headless,
            "web_mode": self.web_mode,
            "web_port": self.web_port,
        }


@dataclass
class DependencyCheck:
    """Result of a dependency check."""
    name: str
    installed: bool
    version: str = ""
    required: bool = True
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "installed": self.installed,
            "version": self.version,
            "required": self.required,
            "error": self.error,
        }


@dataclass
class StartupReport:
    """Report from startup process."""
    success: bool = False
    first_run: bool = False
    platform_info: Dict[str, str] = field(default_factory=dict)
    dependencies: List[DependencyCheck] = field(default_factory=list)
    model_found: bool = False
    model_path: str = ""
    config_generated: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    startup_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "first_run": self.first_run,
            "platform_info": self.platform_info,
            "dependencies": [d.to_dict() for d in self.dependencies],
            "model_found": self.model_found,
            "model_path": self.model_path,
            "config_generated": self.config_generated,
            "errors": self.errors,
            "warnings": self.warnings,
            "startup_time": self.startup_time,
        }


class FirstRunDetector:
    """Detects if this is the first time Rose is being run."""

    def __init__(self, workspace_dir: str = ""):
        self._workspace = Path(workspace_dir) if workspace_dir else Path.cwd() / ".rose"
        self._marker_file = self._workspace / ".first_run_complete"

    def is_first_run(self) -> bool:
        """Check if this is the first run."""
        return not self._marker_file.exists()

    def mark_complete(self):
        """Mark first run as complete."""
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._marker_file.write_text(json.dumps({
            "completed_at": time.time(),
            "platform": platform.system(),
        }))

    def get_state(self) -> Dict[str, Any]:
        return {
            "first_run": self.is_first_run(),
            "workspace": str(self._workspace),
            "marker_exists": self._marker_file.exists(),
        }


class DependencyValidator:
    """Validates that required dependencies are installed."""

    REQUIRED_DEPS = [
        {"name": "llama_cpp", "import_name": "llama_cpp", "required": True},
        {"name": "PIL", "import_name": "PIL", "required": False},
        {"name": "numpy", "import_name": "numpy", "required": False},
    ]

    OPTIONAL_DEPS = [
        {"name": "PySide6", "import_name": "PySide6", "required": False},
        {"name": "playwright", "import_name": "playwright", "required": False},
        {"name": "cv2", "import_name": "cv2", "required": False},
    ]

    def check_all(self) -> List[DependencyCheck]:
        """Check all dependencies."""
        results = []
        for dep in self.REQUIRED_DEPS:
            results.append(self._check_dep(dep))
        for dep in self.OPTIONAL_DEPS:
            results.append(self._check_dep(dep))
        return results

    def _check_dep(self, dep: Dict[str, Any]) -> DependencyCheck:
        """Check a single dependency."""
        import_name = dep["import_name"]
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", getattr(mod, "VERSION", "unknown"))
            if isinstance(version, tuple):
                version = ".".join(str(v) for v in version)
            return DependencyCheck(
                name=dep["name"],
                installed=True,
                version=str(version),
                required=dep.get("required", False),
            )
        except ImportError as e:
            return DependencyCheck(
                name=dep["name"],
                installed=False,
                required=dep.get("required", False),
                error=str(e),
            )

    def all_required_installed(self) -> bool:
        """Check if all required dependencies are installed."""
        for dep in self.REQUIRED_DEPS:
            check = self._check_dep(dep)
            if not check.installed:
                return False
        return True


class ModelDiscovery:
    """Discovers and validates local GGUF models."""

    COMMON_MODEL_DIRS = [
        Path.home() / "models",
        Path.home() / ".rose" / "models",
        Path.cwd() / "models",
        Path("C:/Users") / os.getenv("USERNAME", "") / "models",
    ]

    MODEL_EXTENSIONS = [".gguf"]

    def __init__(self, models_dir: str = ""):
        self._models_dir = Path(models_dir) if models_dir else None

    def find_models(self) -> List[Dict[str, Any]]:
        """Find all GGUF models."""
        models = []
        search_dirs = [self._models_dir] if self._models_dir else self.COMMON_MODEL_DIRS

        for d in search_dirs:
            if d and d.exists():
                models.extend(self._scan_directory(d))

        return models

    def find_best_model(self) -> Optional[Dict[str, Any]]:
        """Find the best available model."""
        models = self.find_models()
        if not models:
            return None

        priority_names = ["qwen", "codellama", "llama", "mistral"]
        for priority in priority_names:
            for model in models:
                if priority in model["name"].lower():
                    return model

        return models[0] if models else None

    def _scan_directory(self, directory: Path) -> List[Dict[str, Any]]:
        """Scan a directory for GGUF models."""
        models = []
        try:
            for f in directory.rglob("*.gguf"):
                size_mb = f.stat().st_size / (1024 * 1024)
                models.append({
                    "name": f.stem,
                    "path": str(f),
                    "size_mb": round(size_mb, 2),
                    "directory": str(f.parent),
                })
        except PermissionError:
            pass
        return models

    def validate_model(self, model_path: str) -> Tuple[bool, str]:
        """Validate a model file."""
        path = Path(model_path)
        if not path.exists():
            return False, f"Model not found: {model_path}"
        if path.suffix != ".gguf":
            return False, f"Not a GGUF file: {model_path}"
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb < 10:
            return False, f"Model too small ({size_mb:.1f} MB)"
        return True, f"Valid model ({size_mb:.1f} MB)"


class ConfigGenerator:
    """Generates default configuration files."""

    DEFAULT_CONFIG = {
        "agent": {
            "name": "Rose",
            "version": "1.1.0",
        },
        "llm": {
            "provider": "local",
            "model_path": "",
            "context_length": 4096,
            "gpu_layers": 28,
        },
        "memory": {
            "short_term_limit": 20,
            "long_term_enabled": True,
        },
        "os_control": {
            "enabled": True,
            "mouse_control": False,
            "keyboard_control": False,
            "window_control": False,
        },
        "vision": {
            "enabled": False,
            "provider": "local",
        },
        "browser": {
            "enabled": False,
            "headless": True,
        },
        "web": {
            "host": "127.0.0.1",
            "port": 8080,
        },
        "security": {
            "safe_mode": True,
            "confirm_dangerous": True,
            "audit_log": True,
        },
    }

    def __init__(self, workspace_dir: str = ""):
        self._workspace = Path(workspace_dir) if workspace_dir else Path.cwd() / ".rose"

    def generate_default(self) -> Dict[str, Any]:
        """Generate default configuration."""
        return json.loads(json.dumps(self.DEFAULT_CONFIG))

    def save_config(self, config: Dict[str, Any], filename: str = "config.json") -> str:
        """Save configuration to file."""
        self._workspace.mkdir(parents=True, exist_ok=True)
        config_path = self._workspace / filename
        config_path.write_text(json.dumps(config, indent=2))
        return str(config_path)

    def load_config(self, filename: str = "config.json") -> Optional[Dict[str, Any]]:
        """Load configuration from file."""
        config_path = self._workspace / filename
        if config_path.exists():
            return json.loads(config_path.read_text())
        return None

    def ensure_config(self, filename: str = "config.json") -> Dict[str, Any]:
        """Load existing config or generate default."""
        existing = self.load_config(filename)
        if existing:
            return existing
        config = self.generate_default()
        self.save_config(config, filename)
        return config


class StartupManager:
    """Manages the complete startup process."""

    def __init__(self, config: Optional[StartupConfig] = None):
        self._config = config or StartupConfig()
        self._first_run = FirstRunDetector(self._config.workspace_dir)
        self._dep_validator = DependencyValidator()
        self._model_discovery = ModelDiscovery(self._config.models_dir)
        self._config_generator = ConfigGenerator(self._config.workspace_dir)

    def run_startup(self) -> StartupReport:
        """Run the complete startup process."""
        start_time = time.time()
        report = StartupReport()

        report.first_run = self._first_run.is_first_run()
        report.platform_info = self._get_platform_info()

        if not self._config.skip_dependency_check:
            report.dependencies = self._dep_validator.check_all()
            for dep in report.dependencies:
                if not dep.installed and dep.required:
                    report.errors.append(f"Missing required dependency: {dep.name}")
                elif not dep.installed:
                    report.warnings.append(f"Optional dependency not installed: {dep.name}")

        if not self._config.skip_model_check:
            best_model = self._model_discovery.find_best_model()
            if best_model:
                report.model_found = True
                report.model_path = best_model["path"]
            else:
                report.warnings.append("No GGUF model found")

        try:
            config = self._config_generator.ensure_config()
            report.config_generated = True
        except Exception as e:
            report.errors.append(f"Config generation failed: {e}")

        report.success = len(report.errors) == 0
        report.startup_time = time.time() - start_time

        if report.first_run and report.success:
            self._first_run.mark_complete()

        return report

    def _get_platform_info(self) -> Dict[str, str]:
        """Get platform information."""
        info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        }
        try:
            if platform.system() == "Windows":
                info["gpu"] = self._detect_windows_gpu()
        except Exception:
            info["gpu"] = "unknown"
        return info

    def _detect_windows_gpu(self) -> str:
        """Detect GPU on Windows."""
        try:
            import subprocess
            result = subprocess.run(
                ["wmic", "path", "win32_videocontroller", "get", "name"],
                capture_output=True, text=True, timeout=5,
            )
            lines = [l.strip() for l in result.stdout.split("\n") if l.strip() and l.strip() != "Name"]
            return lines[0] if lines else "unknown"
        except Exception:
            return "unknown"

    def get_quick_status(self) -> Dict[str, Any]:
        """Get quick startup status without full validation."""
        return {
            "first_run": self._first_run.is_first_run(),
            "deps_ok": self._dep_validator.all_required_installed(),
            "models_found": len(self._model_discovery.find_models()),
        }
