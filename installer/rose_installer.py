"""Rose AI Agent - Windows GUI Installer

This installer deploys a pre-built, self-contained Rose.exe to the target machine.
It does NOT create a Python virtual environment or run pip.
All Python dependencies are bundled inside Rose.exe via PyInstaller.
"""
import os
import sys
import tkinter as tk
import threading
import subprocess
import shutil
import time
import winreg
from tkinter import ttk, messagebox, scrolledtext
from pathlib import Path

APP_VERSION = "1.1.0"
APP_NAME = "Rose"
DEFAULT_INSTALL_DIR = r"C:\Program Files\Rose"
USER_DATA_DIR = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Rose")
INSTALLER_LOG = os.path.join(USER_DATA_DIR, "Installer", "installer.log")

ACCENT = "#e94560"; BG_DARK = "#1a1a2e"; BG_MID = "#16213e"; BG_LIGHT = "#0f3460"
FG = "#eaeaea"; FG_DIM = "#8899aa"; FG_BRIGHT = "#ffffff"
PASS_COLOR = "#2ecc71"; WARN_COLOR = "#f39c12"; FAIL_COLOR = "#e74c3c"


def _find_rose_exe():
    """Locate the pre-built Rose.exe for deployment."""
    # When running as the installer (Inno Setup), the files are in the temp extract dir
    # When running standalone, check common locations
    candidates = [
        Path(__file__).parent.parent / "dist" / "Rose" / "Rose.exe",
        Path(__file__).parent.parent / "dist" / "Rose.exe",
        Path(os.environ.get("TEMP", "")) / "Rose" / "Rose.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _find_models_dir():
    """Locate pre-downloaded model files for deployment."""
    candidates = [
        Path(__file__).parent.parent / "models",
        Path(__file__).parent.parent.parent / "models",
    ]
    for c in candidates:
        if c.exists() and any(c.glob("*.gguf")):
            return c
    return None


class InstallerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} Installer v{APP_VERSION}")
        self.geometry("680x520")
        self.configure(bg=BG_DARK)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self.install_dir = tk.StringVar(value=DEFAULT_INSTALL_DIR)
        self.data_dir = tk.StringVar(value=USER_DATA_DIR)
        self.create_shortcut = tk.BooleanVar(value=True)
        self.cancelled = False
        self.pages = {}
        self.page_order = ["welcome", "license", "location", "system", "install", "finish"]
        self.current_idx = 0

        self._build_nav()
        self._build_pages()
        self.show_page("welcome")

    def _build_nav(self):
        nav = tk.Frame(self, bg=BG_MID, height=50)
        nav.pack(fill=tk.X, side=tk.BOTTOM)
        nav.pack_propagate(False)
        self.btn_back = tk.Button(nav, text="< Back", command=self._prev_page,
                                  bg=BG_LIGHT, fg=FG, activebackground=ACCENT, relief=tk.FLAT, padx=15, pady=6,
                                  font=("Segoe UI", 10))
        self.btn_back.pack(side=tk.LEFT, padx=10, pady=10)
        self.btn_next = tk.Button(nav, text="Next >", command=self._next_page,
                                  bg=ACCENT, fg=FG_BRIGHT, activebackground="#c0392b", relief=tk.FLAT, padx=15, pady=6,
                                  font=("Segoe UI", 10, "bold"))
        self.btn_next.pack(side=tk.RIGHT, padx=10, pady=10)
        self.btn_cancel = tk.Button(nav, text="Cancel", command=self._on_cancel,
                                    bg=BG_LIGHT, fg=FG, activebackground=FAIL_COLOR, relief=tk.FLAT, padx=15, pady=6,
                                    font=("Segoe UI", 10))
        self.btn_cancel.pack(side=tk.RIGHT, padx=5, pady=10)

    def _build_pages(self):
        container = tk.Frame(self, bg=BG_DARK)
        container.pack(fill=tk.BOTH, expand=True)
        self.pages["welcome"] = self._page_welcome(container)
        self.pages["license"] = self._page_license(container)
        self.pages["location"] = self._page_location(container)
        self.pages["system"] = self._page_system(container)
        self.pages["install"] = self._page_install(container)
        self.pages["finish"] = self._page_finish(container)

    def show_page(self, name):
        for p in self.pages.values():
            p.pack_forget()
        self.pages[name].pack(fill=tk.BOTH, expand=True)
        idx = self.page_order.index(name)
        self.current_idx = idx
        self.btn_back.config(state=tk.NORMAL if idx > 0 and name != "install" else tk.DISABLED)
        if name == "install":
            self.btn_next.config(state=tk.DISABLED)
            self.btn_cancel.config(state=tk.DISABLED)
            self._start_install()
        elif name == "finish":
            self.btn_next.config(text="Finish", command=self._finish)
            self.btn_back.config(state=tk.DISABLED)
            self.btn_cancel.config(state=tk.DISABLED)
        else:
            self.btn_next.config(text="Next >", command=self._next_page, state=tk.NORMAL)
            self.btn_cancel.config(state=tk.NORMAL)
        self._update_license_state()

    def _next_page(self):
        if self.current_idx < len(self.page_order) - 1:
            self.show_page(self.page_order[self.current_idx + 1])

    def _prev_page(self):
        if self.current_idx > 0:
            self.show_page(self.page_order[self.current_idx - 1])

    def _on_cancel(self):
        if messagebox.askyesno("Cancel", "Are you sure you want to cancel the installation?"):
            self.cancelled = True
            self.destroy()

    def _finish(self):
        self.destroy()

    def _update_license_state(self):
        pass

    # ─── Welcome Page ────────────────────────────────────────────────────
    def _page_welcome(self, parent):
        f = tk.Frame(parent, bg=BG_DARK)
        tk.Label(f, text="ROSE", font=("Segoe UI", 40, "bold"), fg=ACCENT, bg=BG_DARK).pack(pady=(30, 5))
        tk.Label(f, text="Autonomous Local AI Agent", font=("Segoe UI", 14), fg=FG_DIM, bg=BG_DARK).pack()
        tk.Frame(f, bg=ACCENT, height=2).pack(fill=tk.X, padx=60, pady=20)
        tk.Label(f, text=f"Version {APP_VERSION}", font=("Segoe UI", 10), fg=FG_DIM, bg=BG_DARK).pack(anchor=tk.W, padx=60)
        for line in [
            "Rose is a fully local autonomous AI agent powered by Qwen2.5-VL.",
            "",
            "This installer will:",
            "  - Check system compatibility",
            "  - Detect GPU / CPU",
            "  - Install Rose application",
            "  - Download AI model (~5.6 GB)",
            "  - Create desktop shortcut",
        ]:
            tk.Label(f, text=line, font=("Segoe UI", 11), fg=FG, bg=BG_DARK, anchor=tk.W).pack(anchor=tk.W, padx=60, pady=1)
        tk.Label(f, text="No Python, Git, or terminal required.", font=("Segoe UI", 10, "italic"),
                 fg=PASS_COLOR, bg=BG_DARK).pack(anchor=tk.W, padx=60, pady=(15, 0))
        return f

    # ─── License Page ────────────────────────────────────────────────────
    def _page_license(self, parent):
        f = tk.Frame(parent, bg=BG_DARK)
        tk.Label(f, text="Terms and Conditions", font=("Segoe UI", 16, "bold"), fg=FG, bg=BG_DARK).pack(anchor=tk.W, padx=30, pady=(15, 10))
        tf = tk.Frame(f, bg="#2a2a4a")
        tf.pack(fill=tk.BOTH, expand=True, padx=30, pady=(0, 10))
        self.license_text = scrolledtext.ScrolledText(tf, wrap=tk.WORD, bg=BG_MID, fg=FG,
            font=("Consolas", 10), relief=tk.FLAT, insertbackground=FG, height=14)
        self.license_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        lic_path = Path(__file__).parent.parent / "LICENSE"
        lic = lic_path.read_text(encoding="utf-8") if lic_path.exists() else "MIT License\n\nCopyright (c) 2025 Devansh Gupta"
        self.license_text.insert(tk.END, lic)
        self.license_text.config(state=tk.DISABLED)
        self.license_accepted = tk.BooleanVar(value=False)
        self.license_cb = tk.Checkbutton(f, text="I accept the Terms and Conditions",
            variable=self.license_accepted, bg=BG_DARK, fg=FG, selectcolor=BG_MID,
            activebackground=BG_DARK, activeforeground=FG, font=("Segoe UI", 11),
            command=self._update_license_state)
        self.license_cb.pack(anchor=tk.W, padx=30, pady=5)
        return f

    def _update_license_state(self):
        if hasattr(self, 'license_accepted'):
            self.btn_next.config(state=tk.NORMAL if self.license_accepted.get() else tk.DISABLED)

    # ─── Location Page ───────────────────────────────────────────────────
    def _page_location(self, parent):
        f = tk.Frame(parent, bg=BG_DARK)
        tk.Label(f, text="Installation Location", font=("Segoe UI", 16, "bold"), fg=FG, bg=BG_DARK).pack(anchor=tk.W, padx=30, pady=(15, 15))
        tk.Label(f, text="Installation Folder:", font=("Segoe UI", 11), fg=FG, bg=BG_DARK).pack(anchor=tk.W, padx=30)
        ef = tk.Frame(f, bg="#2a2a4a")
        ef.pack(fill=tk.X, padx=30, pady=(5, 15))
        tk.Entry(ef, textvariable=self.install_dir, font=("Consolas", 11), bg=BG_MID, fg=FG, insertbackground=FG, relief=tk.FLAT, bd=5).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(ef, text="Browse...", command=self._browse, bg=BG_LIGHT, fg=FG, relief=tk.FLAT, padx=12, pady=4, font=("Segoe UI", 10)).pack(side=tk.RIGHT, padx=5, pady=5)
        tk.Label(f, text="User Data Location:", font=("Segoe UI", 11), fg=FG, bg=BG_DARK).pack(anchor=tk.W, padx=30)
        tk.Label(f, text=self.data_dir.get(), font=("Consolas", 10), fg=FG_DIM, bg=BG_DARK).pack(anchor=tk.W, padx=45, pady=(2, 15))
        tk.Label(f, text="Logs, sessions, and workspace data are stored separately.", font=("Segoe UI", 9), fg=FG_DIM, bg=BG_DARK).pack(anchor=tk.W, padx=45)
        tk.Checkbutton(f, text="Create desktop shortcut", variable=self.create_shortcut, bg=BG_DARK, fg=FG, selectcolor=BG_MID, activebackground=BG_DARK, font=("Segoe UI", 11)).pack(anchor=tk.W, padx=30, pady=(20, 5))
        nf = tk.Frame(f, bg=BG_LIGHT)
        nf.pack(fill=tk.X, padx=30, pady=(15, 0))
        tk.Label(nf, text="Internet Required", font=("Segoe UI", 11, "bold"), fg=WARN_COLOR, bg=BG_LIGHT).pack(anchor=tk.W, padx=15, pady=(8, 3))
        tk.Label(nf, text="Rose will download ~5.6 GB of AI model data during installation.", font=("Segoe UI", 10), fg=FG, bg=BG_LIGHT).pack(anchor=tk.W, padx=15, pady=(0, 8))
        return f

    def _browse(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(initialdir=self.install_dir.get())
        if d:
            self.install_dir.set(d)

    # ─── System Check Page ───────────────────────────────────────────────
    def _page_system(self, parent):
        f = tk.Frame(parent, bg=BG_DARK)
        tk.Label(f, text="System Compatibility", font=("Segoe UI", 16, "bold"), fg=FG, bg=BG_DARK).pack(anchor=tk.W, padx=30, pady=(15, 10))
        self.check_frame = tk.Frame(f, bg=BG_DARK)
        self.check_frame.pack(fill=tk.BOTH, expand=True, padx=30)
        self.check_loading = tk.Label(self.check_frame, text="Checking system...", font=("Segoe UI", 12), fg=FG_DIM, bg=BG_DARK)
        self.check_loading.pack(pady=20)
        self.system_ok = False
        return f

    def _run_system_check(self):
        self.check_loading.pack(pady=20)
        threading.Thread(target=self._do_system_check, daemon=True).start()

    def _do_system_check(self):
        import platform, ctypes
        results = []
        # Windows
        wp = platform.platform()
        results.append(("Windows", wp, "pass"))
        # Architecture
        arch = platform.machine()
        results.append(("Architecture", arch, "pass" if "64" in arch or "AMD64" in arch else "warn"))
        # CPU
        try:
            import multiprocessing
            cores = multiprocessing.cpu_count()
        except Exception:
            cores = 0
        results.append(("CPU", f"{cores} cores", "pass" if cores >= 4 else "warn"))
        # RAM
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            mem = MEMORYSTATUSEX(); mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            ram_gb = round(mem.ullTotalPhys / (1024**3), 1)
        except Exception:
            ram_gb = 0
        results.append(("RAM", f"{ram_gb} GB", "pass" if ram_gb >= 8 else "warn"))
        # Disk
        try:
            install_drive = self.install_dir.get()[:3]
            free = ctypes.c_ulonglong(0); total = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(install_drive), None, ctypes.pointer(total), ctypes.pointer(free))
            free_gb = round(free.value / (1024**3), 1)
        except Exception:
            free_gb = 0
        results.append(("Disk Space", f"{free_gb} GB free", "pass" if free_gb >= 10 else "fail"))
        # GPU
        gpu_name = "Not detected"
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                parts = r.stdout.strip().split(", ")
                gpu_name = parts[0].strip()
                vram = int(parts[1].strip()) // 1024 if len(parts) > 1 else 0
                results.append(("NVIDIA GPU", f"{gpu_name} ({vram} GB)", "pass"))
            else:
                results.append(("NVIDIA GPU", "Not detected (CPU mode)", "warn"))
        except Exception:
            results.append(("NVIDIA GPU", "Not detected (CPU mode)", "warn"))
        # Internet
        try:
            import urllib.request
            urllib.request.urlopen("https://www.google.com", timeout=5)
            internet = True
        except Exception:
            internet = False
        results.append(("Internet", "Connected" if internet else "Not connected", "pass" if internet else "fail"))
        self.after(0, lambda: self._display_checks(results, free_gb, internet))

    def _display_checks(self, checks, free_gb, internet):
        self.check_loading.pack_forget()
        for w in self.check_frame.winfo_children():
            w.destroy()
        for label, detail, status in checks:
            row = tk.Frame(self.check_frame, bg=BG_DARK)
            row.pack(fill=tk.X, pady=2)
            icon = {"pass": "\u2713", "warn": "\u26A0", "fail": "\u2717"}[status]
            color = {"pass": PASS_COLOR, "warn": WARN_COLOR, "fail": FAIL_COLOR}[status]
            tk.Label(row, text=icon, font=("Segoe UI", 12), fg=color, bg=BG_DARK, width=3).pack(side=tk.LEFT)
            tk.Label(row, text=label, font=("Segoe UI", 11, "bold"), fg=FG, bg=BG_DARK).pack(side=tk.LEFT, padx=(0, 10))
            tk.Label(row, text=detail, font=("Segoe UI", 10), fg=FG_DIM, bg=BG_DARK).pack(side=tk.LEFT)
        # Disk summary
        sep = tk.Frame(self.check_frame, bg="#2a2a4a", height=1)
        sep.pack(fill=tk.X, pady=8)
        required = 10
        sf = tk.Frame(self.check_frame, bg=BG_DARK)
        sf.pack(fill=tk.X)
        tk.Label(sf, text=f"Required: ~{required} GB  |  Available: {free_gb} GB", font=("Segoe UI", 10), fg=FG, bg=BG_DARK).pack(anchor=tk.W)
        if free_gb >= required:
            tk.Label(sf, text="\u2713 Sufficient disk space", font=("Segoe UI", 10, "bold"), fg=PASS_COLOR, bg=BG_DARK).pack(anchor=tk.W)
        else:
            tk.Label(sf, text=f"\u2717 Insufficient disk space (need ~{required - free_gb:.1f} GB more)", font=("Segoe UI", 10, "bold"), fg=FAIL_COLOR, bg=BG_DARK).pack(anchor=tk.W)
        self.system_ok = internet and free_gb >= 5
        self.btn_next.config(state=tk.NORMAL if self.system_ok else tk.DISABLED)

    # ─── Install Progress Page ───────────────────────────────────────────
    def _page_install(self, parent):
        f = tk.Frame(parent, bg=BG_DARK)
        self.inst_title = tk.Label(f, text="Installing Rose", font=("Segoe UI", 16, "bold"), fg=FG, bg=BG_DARK)
        self.inst_title.pack(anchor=tk.W, padx=30, pady=(15, 10))
        self.inst_step = tk.Label(f, text="Preparing...", font=("Segoe UI", 11), fg=FG_DIM, bg=BG_DARK)
        self.inst_step.pack(anchor=tk.W, padx=30)
        self.inst_overall = ttk.Progressbar(f, length=580, mode="determinate")
        self.inst_overall.pack(fill=tk.X, padx=30, pady=(5, 3))
        self.inst_pct = tk.Label(f, text="0%", font=("Segoe UI", 10), fg=FG, bg=BG_DARK)
        self.inst_pct.pack(anchor=tk.W, padx=30)
        self.inst_file_label = tk.Label(f, text="", font=("Segoe UI", 10), fg=FG, bg=BG_DARK)
        self.inst_file_label.pack(anchor=tk.W, padx=30, pady=(8, 0))
        self.inst_file = ttk.Progressbar(f, length=580, mode="determinate")
        self.inst_file.pack(fill=tk.X, padx=30, pady=(3, 0))
        df = tk.Frame(f, bg=BG_DARK)
        df.pack(fill=tk.X, padx=30, pady=5)
        self.inst_size = tk.Label(df, text="", font=("Consolas", 10), fg=FG_DIM, bg=BG_DARK)
        self.inst_size.pack(anchor=tk.W)
        self.inst_speed = tk.Label(df, text="", font=("Consolas", 10), fg=FG_DIM, bg=BG_DARK)
        self.inst_speed.pack(anchor=tk.W)
        self.inst_eta = tk.Label(df, text="", font=("Consolas", 10), fg=FG_DIM, bg=BG_DARK)
        self.inst_eta.pack(anchor=tk.W)
        lf = tk.Frame(f, bg=BG_DARK)
        lf.pack(fill=tk.BOTH, expand=True, padx=30, pady=(5, 0))
        self.inst_log = scrolledtext.ScrolledText(lf, wrap=tk.WORD, bg=BG_MID, fg=FG_DIM, font=("Consolas", 9), relief=tk.FLAT, height=7)
        self.inst_log.pack(fill=tk.BOTH, expand=True)
        self.install_complete = False
        self.install_error = None
        return f

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.inst_log.insert(tk.END, f"[{ts}] {msg}\n")
        self.inst_log.see(tk.END)

    def _set_step(self, text):
        self.after(0, lambda: self.inst_step.config(text=text))

    def _set_overall(self, pct):
        self.after(0, lambda: (self.inst_overall.config(value=pct), self.inst_pct.config(text=f"{int(pct)}%")))

    def _set_file(self, name, pct=0):
        self.after(0, lambda: (self.inst_file_label.config(text=name), self.inst_file.config(value=pct)))

    def _set_stats(self, size="", speed="", eta=""):
        self.after(0, lambda: (self.inst_size.config(text=size), self.inst_speed.config(text=speed), self.inst_eta.config(text=eta)))

    def _start_install(self):
        threading.Thread(target=self._run_install, daemon=True).start()

    def _run_install(self):
        try:
            install_dir = self.install_dir.get()
            data_dir = self.data_dir.get()
            models_dir = os.path.join(data_dir, "models")

            self.after(0, lambda: self._log("Starting installation..."))
            self._set_step("Step 1/4: Creating directories...")
            self._set_overall(5)
            os.makedirs(install_dir, exist_ok=True)
            for d in ["models", "configs", "scripts"]:
                os.makedirs(os.path.join(install_dir, d), exist_ok=True)
            for d in ["logs", "data", "sessions", "workspace", "models", "cache"]:
                os.makedirs(os.path.join(data_dir, d), exist_ok=True)
            self.after(0, lambda: self._log(f"Install dir: {install_dir}"))
            self.after(0, lambda: self._log(f"Data dir: {data_dir}"))

            # Step 2: Copy Rose.exe and supporting files
            self._set_step("Step 2/4: Installing Rose application...")
            self._set_overall(15)
            self._copy_application(install_dir)

            # Step 3: Download models
            self._set_step("Step 3/4: Downloading AI model...")
            self._set_overall(30)
            self._download_models(models_dir)

            # Step 4: Configure and create shortcuts
            self._set_step("Step 4/4: Configuring Rose...")
            self._set_overall(85)
            self._configure(install_dir, data_dir, models_dir)

            self._set_overall(100)
            self._set_step("Installation complete!")
            self.after(0, lambda: self._log("Installation completed successfully!"))
            self.install_complete = True
            self.after(0, lambda: self.btn_next.config(state=tk.NORMAL))

        except Exception as e:
            self.install_error = str(e)
            self.after(0, lambda: self._log(f"ERROR: {e}"))
            self._set_step(f"Installation failed: {e}")
            self.after(0, lambda: self.btn_next.config(state=tk.NORMAL))

    def _copy_application(self, install_dir):
        """Copy pre-built Rose.exe and supporting files to install directory."""
        self.after(0, lambda: self._log("Copying application files..."))

        # Find Rose.exe - check multiple locations
        rose_exe = None
        search_paths = [
            Path(__file__).parent.parent / "dist" / "Rose" / "Rose.exe",
            Path(__file__).parent.parent / "dist" / "Rose.exe",
            Path(os.path.dirname(sys.argv[0])) / "Rose.exe" if sys.argv else None,
            Path(os.environ.get("TEMP", "")) / "Rose" / "Rose.exe",
        ]
        for p in search_paths:
            if p and p.exists():
                rose_exe = p
                break

        if rose_exe:
            shutil.copy2(rose_exe, os.path.join(install_dir, "Rose.exe"))
            self.after(0, lambda: self._log(f"Copied Rose.exe ({rose_exe.stat().st_size // (1024*1024)} MB)"))
        else:
            # Rose.exe not found - this is a development install, copy source files
            self.after(0, lambda: self._log("Rose.exe not found, installing from source..."))
            src_dir = Path(__file__).parent.parent
            items = ["agent", "scripts", "configs", "run.py", "requirements-runtime.txt",
                     "pyproject.toml", "README.md", "LICENSE"]
            for item in items:
                src = src_dir / item
                dst = Path(install_dir) / item
                if src.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                elif src.is_file():
                    shutil.copy2(src, dst)
            self.after(0, lambda: self._log("Source files copied."))

        # Copy bundled _internal directory if present (PyInstaller output)
        rose_dir = Path(__file__).parent.parent / "dist" / "Rose"
        internal = rose_dir / "_internal"
        if internal.exists():
            dst_internal = os.path.join(install_dir, "_internal")
            if os.path.exists(dst_internal):
                shutil.rmtree(dst_internal)
            shutil.copytree(str(internal), dst_internal)
            self.after(0, lambda: self._log("Copied runtime bundle."))

    def _download_models(self, models_dir):
        """Download AI model files with progress."""
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from model_downloader import ModelDownloader, QWEN_MODEL_FILES, format_bytes, format_speed, format_eta
        except ImportError:
            self.after(0, lambda: self._log("Model downloader not available. Models can be downloaded later."))
            return

        dl = ModelDownloader(models_dir)
        total = dl.get_total_size()
        self.after(0, lambda: self._log(f"Downloading models ({format_bytes(total)})..."))

        existing = dl.get_downloaded_size()
        if existing >= total * 0.95:
            self.after(0, lambda: self._log("Models already present, skipping download."))
            self._set_overall(80)
            return

        def on_progress(progress):
            if progress.status == "downloading" and progress.total_bytes > 0:
                pct = int(progress.bytes_downloaded / progress.total_bytes * 100)
                self._set_file(f"{progress.filename}: {format_bytes(progress.bytes_downloaded)} / {format_bytes(progress.total_bytes)}", pct)
                if progress.speed_bps > 0:
                    self._set_stats(
                        f"{format_bytes(progress.bytes_downloaded)} / {format_bytes(progress.total_bytes)}",
                        f"Speed: {format_speed(progress.speed_bps)}",
                        f"ETA: {format_eta(progress.eta_seconds)}"
                    )
                overall = 30 + (pct * 50 / 100)
                self._set_overall(overall)
            elif progress.status == "complete":
                self._set_file("Download complete", 100)
                self._set_overall(80)
            elif progress.status == "error":
                self.after(0, lambda: self._log(f"Download error: {progress.error_message}"))

        dl.set_progress_callback(on_progress)
        if not dl.download_all():
            if not dl._cancelled:
                raise Exception("Model download failed. Check internet connection.")

    def _configure(self, install_dir, data_dir, models_dir):
        """Write configuration and create shortcuts."""
        self.after(0, lambda: self._log("Writing configuration..."))
        rose_exe = os.path.join(install_dir, "Rose.exe")
        use_exe = os.path.exists(rose_exe)

        # Write .env
        env_path = os.path.join(install_dir, ".env")
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(f"ROSE_INSTALL_DIR={install_dir}\nROSE_DATA_DIR={data_dir}\n"
                    f"ROSE_MODELS_DIR={models_dir}\nROSE_VERSION={APP_VERSION}\n")

        if self.create_shortcut.get():
            self.after(0, lambda: self._log("Creating shortcuts..."))
            self._create_desktop_shortcut(install_dir, use_exe)
            self._create_start_menu(install_dir, use_exe)

        self._register_uninstall(install_dir, data_dir)
        self.after(0, lambda: self._log("Configuration complete."))

    def _create_desktop_shortcut(self, install_dir, use_exe):
        try:
            import win32com.client
            shell = win32com.client.Dispatch("WScript.Shell")
            lnk = os.path.join(shell.SpecialFolders("Desktop"), f"{APP_NAME}.lnk")
            s = shell.CreateShortCut(lnk)
            if use_exe:
                s.Targetpath = os.path.join(install_dir, "Rose.exe")
            else:
                s.Targetpath = os.path.join(install_dir, "runtime", "Scripts", "python.exe")
                s.Arguments = f'"{os.path.join(install_dir, "run.py")}"'
            s.WorkingDirectory = install_dir
            s.IconLocation = os.path.join(install_dir, "Rose.exe") if use_exe else ""
            s.Description = APP_NAME
            s.save()
            self.after(0, lambda: self._log(f"Desktop shortcut: {lnk}"))
        except ImportError:
            # Fallback: create .lnk manually
            desktop = os.path.join(os.environ.get("USERPROFILE", ""), "Desktop", f"{APP_NAME}.lnk")
            self._write_lnk(desktop, install_dir, use_exe)

    def _create_start_menu(self, install_dir, use_exe):
        try:
            import win32com.client
            shell = win32com.client.Dispatch("WScript.Shell")
            folder = os.path.join(shell.SpecialFolders("Programs"), APP_NAME)
            os.makedirs(folder, exist_ok=True)
            lnk = os.path.join(folder, f"{APP_NAME}.lnk")
            s = shell.CreateShortCut(lnk)
            if use_exe:
                s.Targetpath = os.path.join(install_dir, "Rose.exe")
            else:
                s.Targetpath = os.path.join(install_dir, "runtime", "Scripts", "python.exe")
                s.Arguments = f'"{os.path.join(install_dir, "run.py")}"'
            s.WorkingDirectory = install_dir
            s.save()
            self.after(0, lambda: self._log(f"Start Menu: {lnk}"))
        except ImportError:
            pass

    def _write_lnk(self, lnk_path, install_dir, use_exe):
        """Write a minimal .lnk file."""
        os.makedirs(os.path.dirname(lnk_path), exist_ok=True)
        if use_exe:
            target = os.path.join(install_dir, "Rose.exe")
        else:
            target = os.path.join(install_dir, "runtime", "Scripts", "python.exe")
        try:
            import struct
            with open(lnk_path, "wb") as f:
                header = b"\x4c\x00\x00\x00"  # Header size
                f.write(header)
                f.write(b"\x01\x14\x02\x00\x00\x00\x00\x00\xc0\x00\x00\x00\x00\x00\x00\x46")
                f.write(struct.pack("<I", 0x01))  # HasLinkTargetIDList
                f.write(struct.pack("<I", 0x00))  # Reserved
                f.write(struct.pack("<I", 0x00))  # Icon index
                # Target
                target_bytes = target.encode("utf-8")
                f.write(struct.pack("<I", len(target_bytes)))
                f.write(target_bytes)
                f.write(b"\x00")
        except Exception:
            pass

    def _register_uninstall(self, install_dir, data_dir):
        """Register Rose with Windows Add/Remove Programs."""
        try:
            key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\RoseAI"
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, APP_NAME)
                winreg.SetValueEx(key, "DisplayVersion", 0, winreg.REG_SZ, APP_VERSION)
                winreg.SetValueEx(key, "Publisher", 0, winreg.REG_SZ, "Devansh Gupta")
                winreg.SetValueEx(key, "InstallLocation", 0, winreg.REG_SZ, install_dir)
                winreg.SetValueEx(key, "EstimatedSize", 0, winreg.REG_DWORD, 7000000)
                rose_exe = os.path.join(install_dir, "Rose.exe")
                if os.path.exists(rose_exe):
                    winreg.SetValueEx(key, "DisplayIcon", 0, winreg.REG_SZ, rose_exe)
                winreg.SetValueEx(key, "NoModify", 0, winreg.REG_DWORD, 1)
                winreg.SetValueEx(key, "NoRepair", 0, winreg.REG_DWORD, 1)
                uninstaller = os.path.join(install_dir, "uninstall.exe")
                if os.path.exists(uninstaller):
                    winreg.SetValueEx(key, "UninstallString", 0, winreg.REG_SZ, f'"{uninstaller}"')
            self.after(0, lambda: self._log("Registered with Windows."))
        except Exception as e:
            self.after(0, lambda: self._log(f"Uninstall registration warning: {e}"))

    # ─── Finish Page ─────────────────────────────────────────────────────
    def _page_finish(self, parent):
        f = tk.Frame(parent, bg=BG_DARK)
        self.finish_label = tk.Label(f, text="", font=("Segoe UI", 16, "bold"), fg=FG, bg=BG_DARK)
        self.finish_label.pack(pady=(40, 10))
        self.finish_detail = tk.Label(f, text="", font=("Segoe UI", 11), fg=FG_DIM, bg=BG_DARK, wraplength=550)
        self.finish_detail.pack(pady=5)
        self.launch_var = tk.BooleanVar(value=True)
        tk.Checkbutton(f, text="Launch Rose now", variable=self.launch_var, bg=BG_DARK, fg=FG, selectcolor=BG_MID, activebackground=BG_DARK, font=("Segoe UI", 11)).pack(pady=15)
        return f

    def _show_finish(self):
        if self.install_error:
            self.finish_label.config(text="Installation Failed", fg=FAIL_COLOR)
            self.finish_detail.config(text=f"Error: {self.install_error}\n\nCheck the log for details.")
        else:
            self.finish_label.config(text="Installation Complete", fg=PASS_COLOR)
            self.finish_detail.config(text=f"Rose has been installed to:\n{self.install_dir.get()}\n\nYou can launch Rose from the desktop shortcut.")
        self.show_page("finish")

    def _finish(self):
        if self.launch_var.get() and not self.install_error:
            rose_exe = os.path.join(self.install_dir.get(), "Rose.exe")
            if os.path.exists(rose_exe):
                subprocess.Popen([rose_exe], creationflags=subprocess.DETACHED_PROCESS)
        self.destroy()


def main():
    app = InstallerApp()
    # Trigger system check after welcome
    def _post_welcome():
        app._run_system_check()
    app.after(100, _post_welcome)
    app.mainloop()


if __name__ == "__main__":
    main()
