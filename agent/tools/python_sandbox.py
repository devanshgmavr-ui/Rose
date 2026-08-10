"""Python code execution sandbox - subprocess isolation."""

import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .base import Tool, ToolResult, Permission, ConfirmationLevel


class PythonSandboxTool(Tool):
    """Executes Python code in an isolated subprocess."""

    def __init__(self, workspace_dir: str = "workspace"):
        self._workspace = Path(workspace_dir).resolve()
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._timeout = 30.0
        self._max_output_size = 50000
        self._max_code_size = 100000

    @property
    def name(self) -> str:
        return "python_sandbox"

    @property
    def description(self) -> str:
        return "Execute Python code in a sandboxed subprocess. Code runs with limited permissions."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
            },
            "required": ["code"],
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
                "return_code": {"type": "integer"},
            },
        }

    @property
    def required_permissions(self) -> list:
        return [Permission.CODE_EXECUTE]

    @property
    def confirmation_level(self) -> ConfirmationLevel:
        return ConfirmationLevel.REQUIRE_CONFIRMATION

    @property
    def timeout(self) -> float:
        return self._timeout

    def validate(self, arguments: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        code = arguments.get("code")
        if not code:
            errors.append("Missing required argument: code")
        elif not isinstance(code, str):
            errors.append("Code must be a string")
        elif len(code) > self._max_code_size:
            errors.append(f"Code exceeds max size ({self._max_code_size} bytes)")
        return len(errors) == 0, errors

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        start = time.time()
        code = arguments["code"]

        sandbox_template = """
import sys
import os

_original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__
_blocked = ['subprocess', 'os.system', 'shutil', 'socket', 'ctypes']

def _restricted_import(name, *args, **kwargs):
    if name in _blocked:
        raise ImportError(f"Import '{name}' is not allowed in sandbox")
    return _original_import(name, *args, **kwargs)

try:
    __builtins__.__import__ = _restricted_import
except (AttributeError, TypeError):
    pass

import io
_stdout_capture = io.StringIO()
_stderr_capture = io.StringIO()
_old_stdout = sys.stdout
_old_stderr = sys.stderr
sys.stdout = _stdout_capture
sys.stderr = _stderr_capture

_code_file = sys.argv[1]
with open(_code_file, 'r', encoding='utf-8') as f:
    _code_source = f.read()

_exit_code = 0
try:
    exec(compile(_code_source, '<sandbox>', 'exec'))
except Exception as e:
    _exit_code = 1
    print(f"Error: {type(e).__name__}: {e}", file=_stderr_capture)
finally:
    sys.stdout = _old_stdout
    sys.stderr = _old_stderr

print("===STDOUT===")
print(_stdout_capture.getvalue())
print("===STDERR===")
print(_stderr_capture.getvalue())
print("===EXIT===")
print(_exit_code)
sys.exit(_exit_code)
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(sandbox_template)
            sandbox_path = f.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            code_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, sandbox_path, code_path],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=str(self._workspace),
                env=self._get_restricted_env(),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )

            stdout = result.stdout
            stderr = result.stderr
            truncated = False

            if len(stdout) > self._max_output_size:
                stdout = stdout[: self._max_output_size] + "\n...[truncated]"
                truncated = True
            if len(stderr) > self._max_output_size:
                stderr = stderr[: self._max_output_size] + "\n...[truncated]"
                truncated = True

            exit_code = result.returncode
            output = self._parse_output(stdout)
            stderr_output = self._parse_stderr(stdout)

            return ToolResult(
                success=exit_code == 0,
                tool_name=self.name,
                output=output,
                error=stderr_output if stderr_output else "",
                execution_time=time.time() - start,
                truncated=truncated,
                metadata={"return_code": exit_code},
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Code execution timed out after {self._timeout}s",
                execution_time=time.time() - start,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Execution failed: {str(e)}",
                execution_time=time.time() - start,
            )
        finally:
            try:
                os.unlink(sandbox_path)
            except Exception:
                pass
            try:
                os.unlink(code_path)
            except Exception:
                pass

    def _parse_output(self, raw_output: str) -> str:
        lines = raw_output.split("\n")
        output_parts = []
        capture = False
        for line in lines:
            if line.strip() == "===STDOUT===":
                capture = True
                continue
            elif line.strip() == "===STDERR===":
                capture = False
                continue
            elif line.strip() == "===EXIT===":
                break
            if capture:
                output_parts.append(line)
        return "\n".join(output_parts).strip()

    def _parse_stderr(self, raw_output: str) -> str:
        lines = raw_output.split("\n")
        error_parts = []
        capture = False
        for line in lines:
            if line.strip() == "===STDERR===":
                capture = True
                continue
            elif line.strip() == "===EXIT===":
                break
            if capture:
                error_parts.append(line)
        return "\n".join(error_parts).strip()

    def _get_restricted_env(self) -> dict:
        env = os.environ.copy()
        restricted_path = str(self._workspace)
        env["PATH"] = ""
        env["PYTHONPATH"] = ""
        env["HOME"] = restricted_path
        env["TEMP"] = restricted_path
        env["TMP"] = restricted_path
        return env
