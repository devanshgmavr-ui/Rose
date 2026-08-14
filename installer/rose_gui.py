"""Rose Application GUI - Entry point for Rose.exe (no terminal window).

This is the main application that runs after installation.
When built with PyInstaller, it is fully self-contained.
"""
import os
import sys
import json
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from tkinter.font import Font
import threading
import time
from pathlib import Path

APP_NAME = "Rose"
APP_VERSION = "1.1.0"


def get_app_dir() -> Path:
    """Get the application directory (where Rose.exe or this script lives).

    When built with PyInstaller, sys._MEIPASS points to the temp bundle dir,
    but the executable itself is in a different location. We want the directory
    containing Rose.exe, not the temp extraction dir.
    """
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        # sys.executable is the path to Rose.exe
        return Path(sys.executable).parent
    else:
        # Running as Python script
        return Path(__file__).parent


def get_data_dir() -> Path:
    """Get the user data directory (%LOCALAPPDATA%\\Rose)."""
    local_app = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    data_dir = Path(local_app) / "Rose"
    data_dir.mkdir(parents=True, exist_ok=True)
    # Create standard subdirectories
    for sub in ["logs", "data", "sessions", "workspace", "models", "cache"]:
        (data_dir / sub).mkdir(exist_ok=True)
    return data_dir


class RoseTheme:
    BG = "#1a1a2e"
    BG_LIGHT = "#16213e"
    BG_CARD = "#0f3460"
    ACCENT = "#e94560"
    TEXT = "#ffffff"
    TEXT_DIM = "#a0a0b0"
    TEXT_SUCCESS = "#2ecc71"
    BORDER = "#2a2a4a"


class RoseApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} v{APP_VERSION}")
        self.geometry("800x600")
        self.configure(bg=RoseTheme.BG)
        self.resizable(True, True)

        self.app_dir = get_app_dir()
        self.data_dir = get_data_dir()

        # Set up sys.path so agent modules can be found
        self._setup_paths()

        # Try to set window icon
        try:
            icon_path = self.app_dir / "resources" / "rose.ico"
            if icon_path.exists():
                self.iconbitmap(str(icon_path))
        except Exception:
            pass

        self._build_ui()
        self._start_agent()

    def _setup_paths(self):
        """Configure sys.path for the bundled or development environment."""
        if getattr(sys, 'frozen', False):
            # In PyInstaller bundle, modules are in _internal/
            # Only add _internal - subdirectories break relative imports
            internal = self.app_dir / "_internal"
            if internal.exists() and str(internal) not in sys.path:
                sys.path.insert(0, str(internal))
        else:
            # Development mode
            if str(self.app_dir) not in sys.path:
                sys.path.insert(0, str(self.app_dir))

    def _build_ui(self):
        # Header
        header = tk.Frame(self, bg=RoseTheme.BG_CARD, height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        logo_font = Font(family="Segoe UI", size=20, weight="bold")
        tk.Label(header, text="ROSE", font=logo_font, fg=RoseTheme.ACCENT, bg=RoseTheme.BG_CARD).pack(side=tk.LEFT, padx=20)
        tk.Label(header, text=f"v{APP_VERSION}", font=Font(family="Segoe UI", size=10),
                 fg=RoseTheme.TEXT_DIM, bg=RoseTheme.BG_CARD).pack(side=tk.LEFT, padx=5)

        # Status bar
        self.status_var = tk.StringVar(value="Initializing...")
        status_bar = tk.Frame(self, bg=RoseTheme.BG_LIGHT, height=30)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        status_bar.pack_propagate(False)
        tk.Label(status_bar, textvariable=self.status_var, font=Font(family="Segoe UI", size=9),
                 fg=RoseTheme.TEXT_DIM, bg=RoseTheme.BG_LIGHT).pack(side=tk.LEFT, padx=10)

        # Main area
        main = tk.Frame(self, bg=RoseTheme.BG)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Chat display
        chat_frame = tk.Frame(main, bg=RoseTheme.BG)
        chat_frame.pack(fill=tk.BOTH, expand=True)

        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, wrap=tk.WORD, bg=RoseTheme.BG_LIGHT, fg=RoseTheme.TEXT,
            font=Font(family="Consolas", size=11), relief=tk.FLAT,
            insertbackground=RoseTheme.TEXT, selectbackground=RoseTheme.ACCENT,
            state=tk.DISABLED
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)

        # Input area
        input_frame = tk.Frame(main, bg=RoseTheme.BG)
        input_frame.pack(fill=tk.X, pady=(10, 0))

        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(
            input_frame, textvariable=self.input_var,
            font=Font(family="Consolas", size=11),
            bg=RoseTheme.BG_LIGHT, fg=RoseTheme.TEXT,
            insertbackground=RoseTheme.TEXT, relief=tk.FLAT
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8)
        self.input_entry.bind("<Return>", self._on_send)

        self.send_btn = tk.Button(
            input_frame, text="Send", command=self._on_send,
            bg=RoseTheme.ACCENT, fg=RoseTheme.TEXT,
            activebackground=RoseTheme.ACCENT, activeforeground=RoseTheme.TEXT,
            font=Font(family="Segoe UI", size=10, weight="bold"),
            relief=tk.FLAT, padx=20, pady=8
        )
        self.send_btn.pack(side=tk.RIGHT, padx=(10, 0))

    def _append_message(self, sender: str, message: str, color: str = RoseTheme.TEXT):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{sender}: {message}\n\n", "msg")
        self.chat_display.tag_config("msg", foreground=color, font=Font(family="Consolas", size=11))
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def _on_send(self, event=None):
        message = self.input_var.get().strip()
        if not message:
            return
        self.input_var.set("")
        self._append_message("You", message, RoseTheme.TEXT)
        threading.Thread(target=self._process_message, args=(message,), daemon=True).start()

    def _process_message(self, message: str):
        try:
            self.after(0, lambda: self.status_var.set("Thinking..."))
            if self.agent:
                response = self.agent.chat(message)
                self.after(0, lambda: self._append_message("Rose", response.text, RoseTheme.TEXT_SUCCESS))
            else:
                self.after(0, lambda: self._append_message("Rose",
                    "Agent not initialized. Please check model files in %LOCALAPPDATA%\\Rose\\models\\",
                    RoseTheme.TEXT_DIM))
        except Exception as e:
            self.after(0, lambda: self._append_message("Rose", f"Error: {e}", "#e74c3c"))
        finally:
            self.after(0, lambda: self.status_var.set("Ready"))

    def _start_agent(self):
        self.agent = None
        threading.Thread(target=self._init_agent, daemon=True).start()

    def _init_agent(self):
        try:
            self.after(0, lambda: self.status_var.set("Loading Rose..."))

            # Set environment for model paths
            models_dir = self.data_dir / "models"
            os.environ["ROSE_DATA_DIR"] = str(self.data_dir)
            os.environ["ROSE_MODELS_DIR"] = str(models_dir)
            os.environ["MODEL_PATH"] = str(models_dir / "Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf")
            os.environ["MMPROJ_PATH"] = str(models_dir / "mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf")

            from agent.core.config import Config
            from agent.core.agent import Agent

            config = Config()
            self.agent = Agent(config=config)

            if self.agent.initialize():
                self.after(0, lambda: self._append_message("Rose",
                    f"Hello! I'm Rose, your autonomous AI assistant.\n"
                    f"Powered by Qwen2.5-VL. How can I help you today?\n\n"
                    f"Models: {models_dir}",
                    RoseTheme.TEXT_SUCCESS))
                self.after(0, lambda: self.status_var.set("Ready"))
            else:
                self.after(0, lambda: self._append_message("Rose",
                    "Warning: Agent initialization had issues. Some features may be limited.\n"
                    f"Models directory: {models_dir}\n"
                    "Ensure model files are downloaded to this directory.",
                    "#f39c12"))
                self.after(0, lambda: self.status_var.set("Partial initialization"))
        except Exception as e:
            self.after(0, lambda: self._append_message("Rose",
                f"Failed to initialize: {e}\n"
                "Please ensure model files are present in %LOCALAPPDATA%\\Rose\\models\\",
                "#e74c3c"))
            self.after(0, lambda: self.status_var.set("Initialization failed"))


def main():
    """Entry point for Rose.exe (no console window)."""
    app = RoseApp()
    app.mainloop()


if __name__ == "__main__":
    main()
