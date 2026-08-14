"""Rose AI Agent - Windows GUI Installer"""
import os, sys, tkinter as tk, threading, subprocess, shutil, time, winreg
from tkinter import ttk, messagebox, scrolledtext

APP_VERSION = "1.1.0"
APP_NAME = "Rose AI Agent"
DEFAULT_INSTALL_DIR = r"C:\Program Files\Rose"
USER_DATA_DIR = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Rose")
INSTALLER_LOG = os.path.join(USER_DATA_DIR, "Installer", "installer.log")

ACCENT = "#e94560"; BG_DARK = "#1a1a2e"; BG_MID = "#16213e"; BG_LIGHT = "#0f3460"
FG = "#eaeaea"; FG_DIM = "#8899aa"; FG_BRIGHT = "#ffffff"
PASS_COLOR = "#2ecc71"; WARN_COLOR = "#f39c12"; FAIL_COLOR = "#e74c3c"

FILES_TO_COPY = [
    ("agent", True), ("scripts", True), ("configs", True), ("run.py", False),
    ("requirements-runtime.txt", False), ("requirements-dev.txt", False),
    ("pyproject.toml", False), ("README.md", False), ("LICENSE", False),
]

LICENSE_TEXT = """MIT License

Copyright (c) 2024 Rose AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""


def _log_write(msg, level="INFO"):
    try:
        os.makedirs(os.path.dirname(INSTALLER_LOG), exist_ok=True)
        with open(INSTALLER_LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{level}] {msg}\n")
    except Exception:
        pass


def log(msg):
    print(msg)
    _log_write(msg)


def log_err(msg):
    _log_write(msg, "ERROR")


def _cb_checkbtn(var, btn, state_true="normal", state_false="disabled"):
    btn.config(state=state_true if var.get() else state_false)


class RoseInstaller(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} Installer v{APP_VERSION}")
        self.geometry("700x520"); self.resizable(False, False)
        self.configure(bg=BG_DARK)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self.install_dir = tk.StringVar(value=DEFAULT_INSTALL_DIR)
        self.data_dir = tk.StringVar(value=USER_DATA_DIR)
        self.create_shortcut = tk.BooleanVar(value=True)
        self.launch_after = tk.BooleanVar(value=True)
        self.license_accepted = tk.BooleanVar(value=False)
        self.cancelled = False
        self.install_thread = None
        self.system_info = None; self.check_results = None

        self._build_styles()
        self.pages = {}
        self._build_nav()
        self._build_pages()
        self.show_page("welcome")

    def _build_styles(self):
        s = ttk.Style(self); s.theme_use("clam")
        s.configure(".", background=BG_DARK, foreground=FG, font=("Segoe UI", 10))
        s.configure("TFrame", background=BG_DARK)
        s.configure("TLabel", background=BG_DARK, foreground=FG)
        s.configure("Title.TLabel", font=("Segoe UI", 20, "bold"), foreground=FG_BRIGHT)
        s.configure("Sub.TLabel", font=("Segoe UI", 11), foreground=FG_DIM)
        s.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), background=ACCENT, foreground=FG_BRIGHT)
        s.map("Accent.TButton", background=[("active", "#ff6b81")])
        s.configure("TButton", font=("Segoe UI", 10))
        s.configure("TEntry", fieldbackground=BG_MID, foreground=FG)
        s.configure(".Horizontal.TProgressbar", troughcolor=BG_MID, background=ACCENT)

    def _build_nav(self):
        nav = tk.Frame(self, bg=BG_MID, height=50)
        nav.pack(side="bottom", fill="x"); nav.pack_propagate(False)
        self.btn_back = ttk.Button(nav, text="< Back", command=self._go_back, width=10)
        self.btn_back.pack(side="left", padx=10, pady=10)
        self.btn_cancel = ttk.Button(nav, text="Cancel", command=self._on_cancel, width=10)
        self.btn_cancel.pack(side="right", padx=10, pady=10)
        self.btn_next = ttk.Button(nav, text="Next >", command=self._go_next, width=10)
        self.btn_next.pack(side="right", padx=5, pady=10)

    def _build_pages(self):
        self.content_frame = tk.Frame(self, bg=BG_DARK)
        self.content_frame.pack(fill="both", expand=True)
        for name, cls in [("welcome", WelcomePage), ("license", LicensePage),
                          ("location", LocationPage), ("syscheck", SystemCheckPage),
                          ("install", InstallPage), ("finish", FinishPage)]:
            self.pages[name] = cls(self.content_frame, self)

    @property
    def _page_order(self):
        return ["welcome", "license", "location", "syscheck", "install", "finish"]

    def _current_page(self):
        for name in self._page_order:
            if self.pages[name].frame.winfo_ismapped():
                return name
        return "welcome"

    def show_page(self, name):
        for p in self.pages.values():
            p.frame.pack_forget()
        self.pages[name].frame.pack(fill="both", expand=True)
        self._update_nav(name)

    def _update_nav(self, name):
        order = self._page_order; idx = order.index(name)
        self.btn_back.config(state="normal" if 0 < idx and name != "install" else "disabled")
        self.btn_cancel.config(state="disabled" if name in ("install", "finish") else "normal")
        special = {"license": lambda: self.license_accepted.get(),
                   "syscheck": lambda: False, "install": lambda: False, "finish": lambda: False}
        if name in special:
            self.btn_next.config(state="normal" if special[name]() else "disabled")
        else:
            self.btn_next.config(state="normal")

    def _go_next(self):
        order = self._page_order; cur = order.index(self._current_page())
        if cur < len(order) - 1:
            nxt = order[cur + 1]
            if nxt == "install":
                self.pages["install"].start_install()
            self.show_page(nxt)

    def _go_back(self):
        order = self._page_order; cur = order.index(self._current_page())
        if cur > 0:
            self.show_page(order[cur - 1])

    def _on_cancel(self):
        if messagebox.askyesno("Cancel", "Are you sure you want to cancel the installation?"):
            self.cancelled = True; self.destroy()


class BasePage:
    def __init__(self, parent, installer):
        self.installer = installer
        self.frame = tk.Frame(parent, bg=BG_DARK)


class WelcomePage(BasePage):
    def __init__(self, parent, installer):
        super().__init__(parent, installer)
        f = self.frame
        tk.Frame(f, bg=BG_DARK, height=60).pack()
        tk.Label(f, text="\U0001f339", font=("Segoe UI Emoji", 56), bg=BG_DARK).pack(pady=(0, 10))
        tk.Label(f, text=APP_NAME, style="Title.TLabel", bg=BG_DARK).pack()
        tk.Label(f, text=f"Version {APP_VERSION}", style="Sub.TLabel", bg=BG_DARK).pack(pady=(2, 20))
        tk.Label(f, text=(
            "Welcome to the Rose AI Agent installer.\n\n"
            "Rose is an advanced AI assistant that runs locally on your machine.\n"
            "This wizard will guide you through the installation process.\n\n"
            "Estimated disk space required: ~5 GB (models downloaded on first run)"
        ), style="Sub.TLabel", bg=BG_DARK, justify="center").pack(pady=10)


class LicensePage(BasePage):
    def __init__(self, parent, installer):
        super().__init__(parent, installer)
        f = self.frame
        tk.Label(f, text="License Agreement", style="Title.TLabel", bg=BG_DARK).pack(anchor="w", padx=30, pady=(20, 5))
        tk.Label(f, text="Please read and accept the license agreement.", style="Sub.TLabel", bg=BG_DARK).pack(anchor="w", padx=30)
        txt = scrolledtext.ScrolledText(f, wrap="word", bg=BG_MID, fg=FG, insertbackground=FG, font=("Consolas", 9), relief="flat")
        txt.pack(fill="both", expand=True, padx=30, pady=10)
        txt.insert("1.0", LICENSE_TEXT); txt.config(state="disabled")
        tk.Checkbutton(f, text="I accept the terms of the license agreement",
                       variable=installer.license_accepted,
                       command=lambda: _cb_checkbtn(installer.license_accepted, installer.btn_next),
                       bg=BG_DARK, fg=FG, selectcolor=BG_MID, activebackground=BG_DARK,
                       activeforeground=FG, font=("Segoe UI", 10)).pack(anchor="w", padx=30, pady=(0, 15))


class LocationPage(BasePage):
    def __init__(self, parent, installer):
        super().__init__(parent, installer)
        f = self.frame
        tk.Label(f, text="Installation Location", style="Title.TLabel", bg=BG_DARK).pack(anchor="w", padx=30, pady=(20, 5))
        tk.Label(f, text="Choose where to install Rose and where to store data.", style="Sub.TLabel", bg=BG_DARK).pack(anchor="w", padx=30)

        box = tk.Frame(f, bg=BG_MID, highlightbackground=BG_LIGHT, highlightthickness=1)
        box.pack(fill="x", padx=30, pady=15)

        for row, (label, var) in enumerate([("Install Directory:", installer.install_dir),
                                             ("User Data Directory:", installer.data_dir)]):
            tk.Label(box, text=label, bg=BG_MID, fg=FG, font=("Segoe UI", 10)).grid(row=row*2, column=0, sticky="w", padx=10, pady=(10 if row==0 else 5, 2))
            rf = tk.Frame(box, bg=BG_MID)
            rf.grid(row=row*2+1, column=0, sticky="ew", padx=10, pady=(0, 10))
            tk.Entry(rf, textvariable=var, bg=BG_DARK, fg=FG, insertbackground=FG,
                     font=("Consolas", 10), relief="flat", width=50).pack(side="left", fill="x", expand=True)
            if row == 0:
                ttk.Button(rf, text="Browse...", width=8, command=lambda v=var: self._browse(v)).pack(side="left", padx=(5, 0))
        box.columnconfigure(0, weight=1)

        tk.Checkbutton(f, text="Create desktop shortcut", variable=installer.create_shortcut,
                       bg=BG_DARK, fg=FG, selectcolor=BG_MID, activebackground=BG_DARK,
                       activeforeground=FG, font=("Segoe UI", 10)).pack(anchor="w", padx=30, pady=(5, 15))

        notice = tk.Frame(f, bg="#2d1b00", highlightbackground=WARN_COLOR, highlightthickness=1)
        notice.pack(fill="x", padx=30, pady=(0, 15))
        tk.Label(notice, text="\u26a0  Internet connection required for initial model download (~2 GB).\n   Models will be cached locally after first download.",
                 bg="#2d1b00", fg=WARN_COLOR, font=("Segoe UI", 9), justify="left").pack(padx=10, pady=10)

    def _browse(self, var):
        from tkinter import filedialog
        d = filedialog.askdirectory(title="Select Directory")
        if d: var.set(d)


class SystemCheckPage(BasePage):
    def __init__(self, parent, installer):
        super().__init__(parent, installer)
        f = self.frame
        tk.Label(f, text="System Check", style="Title.TLabel", bg=BG_DARK).pack(anchor="w", padx=30, pady=(20, 5))
        tk.Label(f, text="Checking your system requirements...", style="Sub.TLabel", bg=BG_DARK).pack(anchor="w", padx=30)
        self.check_frame = tk.Frame(f, bg=BG_MID, highlightbackground=BG_LIGHT, highlightthickness=1)
        self.check_frame.pack(fill="both", expand=True, padx=30, pady=10)
        self.status_label = tk.Label(f, text="Running system checks...", style="Sub.TLabel", bg=BG_DARK)
        self.status_label.pack(anchor="w", padx=30)
        self.rows = {}
        self._build_rows(["Operating System", "CPU", "RAM", "GPU", "Disk Space", "Internet"])
        threading.Thread(target=self._run_checks, daemon=True).start()

    def _build_rows(self, checks):
        for w in self.check_frame.winfo_children(): w.destroy()
        self.rows = {}
        for i, name in enumerate(checks):
            tk.Label(self.check_frame, text=name, bg=BG_MID, fg=FG, font=("Segoe UI", 10)).grid(row=i, column=0, sticky="w", padx=15, pady=8)
            st = tk.Label(self.check_frame, text="Checking...", bg=BG_MID, fg=FG_DIM, font=("Segoe UI", 10))
            st.grid(row=i, column=1, sticky="w", padx=15); self.rows[name] = st
        self.check_frame.columnconfigure(1, weight=1)

    def _run_checks(self):
        try:
            from system_check import get_system_info, get_check_results
            self.system_info = get_system_info()
            results = get_check_results(self.system_info)
            self.installer.check_results = results
            mapping = {"os": "Operating System", "cpu": "CPU", "ram": "RAM", "gpu": "GPU", "disk": "Disk Space", "internet": "Internet"}
            for key, name in mapping.items():
                if key in results:
                    r = results[key]
                    status = r.get("status", "warn") if isinstance(r, dict) else "warn"
                    detail = r.get("detail", "") if isinstance(r, dict) else str(r)
                    self.after(0, self._set_row, name, status, detail)
            self.after(0, self._done)
        except Exception as e:
            log_err(f"System check error: {e}")
            self._fallback()

    def _fallback(self):
        import platform
        checks = [("Operating System", "pass", platform.platform()), ("CPU", "pass", f"{os.cpu_count()} cores"),
                  ("RAM", "warn", "Could not detect"), ("GPU", "warn", "Could not detect"),
                  ("Disk Space", "pass", f"{shutil.disk_usage('C:').free // (1024**3)} GB free"),
                  ("Internet", "warn", "Could not verify")]
        for i, (name, status, detail) in enumerate(checks):
            self.after(i * 150, self._set_row, name, status, detail)
        self.after(len(checks) * 150, self._done)

    def _set_row(self, name, status, detail):
        if name not in self.rows: return
        icons = {"pass": "\u2713", "warn": "\u26a0", "fail": "\u2717"}
        colors = {"pass": PASS_COLOR, "warn": WARN_COLOR, "fail": FAIL_COLOR}
        self.rows[name].config(text=f"{icons.get(status, '?')}  {detail}", fg=colors.get(status, FG_DIM))

    def _done(self):
        self.status_label.config(text="System check complete. Click Next to continue.")
        self.installer.btn_next.config(state="normal")


class InstallPage(BasePage):
    def __init__(self, parent, installer):
        super().__init__(parent, installer)
        self.running = False
        f = self.frame
        tk.Label(f, text="Installing Rose", style="Title.TLabel", bg=BG_DARK).pack(anchor="w", padx=30, pady=(15, 5))

        pf = tk.Frame(f, bg=BG_MID, highlightbackground=BG_LIGHT, highlightthickness=1)
        pf.pack(fill="x", padx=30, pady=(0, 5))

        tk.Label(pf, text="Overall Progress:", bg=BG_MID, fg=FG, font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", padx=10, pady=(8, 2))
        self.overall_bar = ttk.Progressbar(pf, length=550, mode="determinate", style="Horizontal.TProgressbar")
        self.overall_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 5))
        self.step_label = tk.Label(pf, text="", bg=BG_MID, fg=FG, font=("Segoe UI", 9))
        self.step_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 8))

        tk.Label(pf, text="File Progress:", bg=BG_MID, fg=FG, font=("Segoe UI", 9)).grid(row=3, column=0, sticky="w", padx=10, pady=(0, 2))
        self.file_bar = ttk.Progressbar(pf, length=550, mode="determinate", style="Horizontal.TProgressbar")
        self.file_bar.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 5))
        self.file_label = tk.Label(pf, text="", bg=BG_MID, fg=FG, font=("Segoe UI", 9))
        self.file_label.grid(row=5, column=0, sticky="w", padx=10)
        self.speed_label = tk.Label(pf, text="", bg=BG_MID, fg=FG_DIM, font=("Segoe UI", 9))
        self.speed_label.grid(row=5, column=1, sticky="e", padx=10)
        pf.columnconfigure(0, weight=1)

        self.log_area = scrolledtext.ScrolledText(f, wrap="word", bg="#0d1117", fg=FG_DIM, font=("Consolas", 8), relief="flat", state="disabled", height=8)
        self.log_area.pack(fill="both", expand=True, padx=30, pady=(5, 10))

    def start_install(self):
        self.running = True; self.installer.cancelled = False
        self.step_label.config(text="Preparing installation...")
        self.overall_bar["value"] = 0; self.file_bar["value"] = 0
        self.installer.install_thread = threading.Thread(target=self._install, daemon=True)
        self.installer.install_thread.start()

    def _log(self, msg):
        def _do():
            self.log_area.config(state="normal")
            self.log_area.insert("end", f"{time.strftime('%H:%M:%S')} {msg}\n")
            self.log_area.see("end"); self.log_area.config(state="disabled")
        self.after(0, _do); log(msg)

    def _set_step(self, idx, total, name):
        self.after(0, lambda: self.step_label.config(text=f"Step {idx}/{total}: {name}"))
        self.after(0, lambda v=idx / total * 100: setattr(self.overall_bar, 'value', v))

    def _set_file(self, name, progress):
        self.after(0, lambda: self.file_label.config(text=name))
        self.after(0, lambda v=progress: setattr(self.file_bar, 'value', v))

    def _set_speed(self, text):
        self.after(0, lambda: self.speed_label.config(text=text))

    def _install(self):
        steps = [("Creating directories", self._step_dirs), ("Copying files", self._step_copy),
                 ("Installing runtime", self._step_runtime), ("Downloading models", self._step_models),
                 ("Configuring installation", self._step_config)]
        try:
            for i, (name, fn) in enumerate(steps, 1):
                if self.installer.cancelled:
                    self._log("Installation cancelled."); self.after(0, self._finish_cancel); return
                self._set_step(i, len(steps), name)
                self._log(f"--- Step {i}/{len(steps)}: {name} ---"); fn()
                self._log(f"Step {i} complete.")
            self._log("\nInstallation completed successfully!"); self.after(0, self._finish_done)
        except Exception as e:
            log_err(f"Install error: {e}"); self._log(f"Error: {e}")
            self.after(0, lambda: messagebox.showerror("Install Error", str(e)))
            self.after(0, self._finish_cancel)

    def _step_dirs(self):
        for d in [self.installer.install_dir.get(), self.installer.data_dir.get()]:
            os.makedirs(d, exist_ok=True); self._log(f"Created: {d}")
        os.makedirs(os.path.join(self.installer.install_dir.get(), "runtime"), exist_ok=True)
        os.makedirs(os.path.join(self.installer.data_dir.get(), "models"), exist_ok=True)
        os.makedirs(os.path.join(self.installer.data_dir.get(), "Installer"), exist_ok=True)

    def _step_copy(self):
        src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dst = self.installer.install_dir.get()
        for name, is_dir in FILES_TO_COPY:
            s = os.path.join(src, name); d = os.path.join(dst, name)
            if not os.path.exists(s):
                self._log(f"Skipping (not found): {name}"); continue
            if is_dir:
                if os.path.isdir(d): shutil.rmtree(d)
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
            self._log(f"Copied: {name}"); self._set_file(name, 0); time.sleep(0.05)

    def _step_runtime(self):
        install_dir = self.installer.install_dir.get()
        runtime = os.path.join(install_dir, "runtime")
        venv_py = os.path.join(runtime, "Scripts", "python.exe")
        if not os.path.exists(venv_py):
            self._log("Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", runtime], check=True, capture_output=True)
        else:
            self._log("Runtime venv already exists, updating...")
        pip = os.path.join(runtime, "Scripts", "pip.exe")
        req = os.path.join(install_dir, "requirements-runtime.txt")
        if os.path.exists(req):
            self._log("Installing dependencies...")
            cmd = [pip, "install", "-r", req, "--only-binary=:all:",
                   "-i", "https://download.pytorch.org/whl/cu121",
                   "--extra-index-url", "https://pypi.org/simple/"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                self._log(f"pip warning: {result.stderr[:200]}")
            else:
                self._log("Dependencies installed.")

    def _step_models(self):
        try:
            from model_downloader import ModelDownloader, QWEN_MODEL_FILES, format_bytes, format_speed, format_eta
            dl = ModelDownloader(os.path.join(self.installer.data_dir.get(), "models"))
            def progress_cb(file, downloaded, total, speed, eta):
                pct = (downloaded / total * 100) if total else 0
                self._set_file(f"{file} ({pct:.0f}%)", pct)
                self._set_speed(f"{format_bytes(downloaded)}/{format_bytes(total)} | {format_speed(speed)} | ETA: {format_eta(eta)}")
            dl.download_all(progress_cb=progress_cb, cancel_check=lambda: self.installer.cancelled)
            self._log("Models downloaded.")
        except ImportError:
            self._log("Model downloader not available. Skipping model download.")
            self._log("Run `python -m rose.models.download` after installation.")
        except Exception as e:
            self._log(f"Model download warning: {e}")

    def _step_config(self):
        install_dir = self.installer.install_dir.get()
        data_dir = self.installer.data_dir.get()
        models_dir = os.path.join(data_dir, "models")
        env_path = os.path.join(install_dir, ".env")
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(f"ROSE_INSTALL_DIR={install_dir}\nROSE_DATA_DIR={data_dir}\n"
                    f"ROSE_MODELS_DIR={models_dir}\nROSE_VERSION={APP_VERSION}\n")
        self._log(f"Created .env at {env_path}")
        if self.installer.create_shortcut.get():
            self._create_desktop_shortcut(install_dir)
            self._create_start_menu_shortcut(install_dir)
        self._register_uninstall(install_dir, data_dir)
        self._log("Configuration complete.")

    def _create_desktop_shortcut(self, install_dir):
        try:
            import win32com.client
            shell = win32com.client.Dispatch("WScript.Shell")
            lnk = os.path.join(shell.SpecialFolders("Desktop"), f"{APP_NAME}.lnk")
            s = shell.CreateShortCut(lnk)
            s.Targetpath = os.path.join(install_dir, "runtime", "Scripts", "python.exe")
            s.Arguments = f'"{os.path.join(install_dir, "run.py")}"'
            s.WorkingDirectory = install_dir
            s.IconLocation = os.path.join(install_dir, "run.py")
            s.Description = APP_NAME; s.save()
            self._log(f"Desktop shortcut: {lnk}")
        except ImportError:
            self._write_lnk(os.path.join(os.environ.get("USERPROFILE", ""), "Desktop", f"{APP_NAME}.lnk"), install_dir)

    def _create_start_menu_shortcut(self, install_dir):
        try:
            import win32com.client
            shell = win32com.client.Dispatch("WScript.Shell")
            folder = os.path.join(shell.SpecialFolders("Programs"), APP_NAME)
            os.makedirs(folder, exist_ok=True)
            lnk = os.path.join(folder, f"{APP_NAME}.lnk")
            s = shell.CreateShortCut(lnk)
            s.Targetpath = os.path.join(install_dir, "runtime", "Scripts", "python.exe")
            s.Arguments = f'"{os.path.join(install_dir, "run.py")}"'
            s.WorkingDirectory = install_dir; s.save()
            self._log(f"Start Menu shortcut: {lnk}")
        except ImportError:
            pass

    def _write_lnk(self, lnk_path, install_dir):
        os.makedirs(os.path.dirname(lnk_path), exist_ok=True)
        target = os.path.join(install_dir, "runtime", "Scripts", "python.exe")
        try:
            import struct
            with open(lnk_path, "wb") as f:
                f.write(b'\x4c\x00\x00\x00\x01\x14\x02\x00')
                f.write(b'\x00' * 20)
                f.write(struct.pack('<I', 0x20))
                f.write(b'\x00' * 4)
                f.write(target.encode('utf-16-le') + b'\x00\x00')
            self._log(f"Shortcut (fallback): {lnk_path}")
        except Exception as e:
            self._log(f"Shortcut fallback failed: {e}")

    def _register_uninstall(self, install_dir, data_dir):
        try:
            key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\RoseAI"
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                vals = [("DisplayName", APP_NAME), ("DisplayVersion", APP_VERSION),
                        ("Publisher", "Rose AI"), ("InstallLocation", install_dir),
                        ("UninstallString", f'"{os.path.join(install_dir, "runtime", "Scripts", "python.exe")}" '
                                            f'"{os.path.join(install_dir, "scripts", "uninstall.py")}"'),
                        ("DisplayIcon", os.path.join(install_dir, "run.py"))]
                for name, val in vals:
                    winreg.SetValueEx(key, name, 0, winreg.REG_SZ, val)
                for name in ("NoModify", "NoRepair"):
                    winreg.SetValueEx(key, name, 0, winreg.REG_DWORD, 1)
            self._log("Registered uninstaller in Windows registry.")
        except Exception as e:
            self._log(f"Registry warning: {e}")

    def _finish_done(self):
        self.running = False
        self.overall_bar["value"] = 100; self.file_bar["value"] = 100
        self.after(0, lambda: self.installer.show_page("finish"))

    def _finish_cancel(self):
        self.running = False
        self.after(0, lambda: self.installer.btn_next.config(state="disabled"))
        self.after(0, lambda: self.installer.btn_cancel.config(state="normal"))


class FinishPage(BasePage):
    def __init__(self, parent, installer):
        super().__init__(parent, installer)
        f = self.frame
        tk.Frame(f, bg=BG_DARK, height=40).pack()
        tk.Label(f, text="\u2713", font=("Segoe UI", 48), fg=PASS_COLOR, bg=BG_DARK).pack(pady=(0, 10))
        tk.Label(f, text="Installation Complete!", style="Title.TLabel", bg=BG_DARK).pack()
        tk.Label(f, text=f"{APP_NAME} v{APP_VERSION} has been installed successfully.",
                 style="Sub.TLabel", bg=BG_DARK).pack(pady=(5, 20))

        info = tk.Frame(f, bg=BG_MID, highlightbackground=BG_LIGHT, highlightthickness=1)
        info.pack(fill="x", padx=60, pady=(0, 15))
        for i, (k, v) in enumerate([("Install:", installer.install_dir.get()), ("Data:", installer.data_dir.get())]):
            tk.Label(info, text=k, bg=BG_MID, fg=FG_DIM, font=("Segoe UI", 10)).grid(row=i, column=0, sticky="w", padx=10, pady=6)
            tk.Label(info, text=v, bg=BG_MID, fg=FG, font=("Consolas", 9)).grid(row=i, column=1, sticky="w", padx=10)

        tk.Checkbutton(f, text="Launch Rose now", variable=installer.launch_after,
                       bg=BG_DARK, fg=FG, selectcolor=BG_MID, activebackground=BG_DARK,
                       activeforeground=FG, font=("Segoe UI", 10)).pack(pady=(0, 10))

        def finish():
            if installer.launch_after.get():
                install_dir = installer.install_dir.get()
                py = os.path.join(install_dir, "runtime", "Scripts", "python.exe")
                run = os.path.join(install_dir, "run.py")
                if os.path.exists(py) and os.path.exists(run):
                    subprocess.Popen([py, run], cwd=install_dir)
            installer.destroy()

        ttk.Button(f, text="Finish", style="Accent.TButton", command=finish, width=16).pack(pady=10)


def main():
    log(f"Starting {APP_NAME} installer v{APP_VERSION}")
    app = RoseInstaller()
    app.mainloop()


if __name__ == "__main__":
    main()
