"""Tests for Stage 5.3 - Controlled CLI Automation."""

import pytest
from agent.tools.cli_automation import CLIExecutor, CLIExecutorTool, CommandResult, ShellType


class TestCommandResult:
    def test_success(self):
        r = CommandResult(True, "echo hello", stdout="hello\n", exit_code=0)
        assert r.success is True
        assert r.stdout == "hello\n"

    def test_failure(self):
        r = CommandResult(False, "bad cmd", stderr="error", exit_code=1)
        assert r.success is False
        assert r.exit_code == 1

    def test_to_dict(self):
        r = CommandResult(True, "echo test", stdout="test", exit_code=0, execution_time=0.5)
        d = r.to_dict()
        assert d["success"] is True
        assert d["execution_time"] == 0.5

    def test_to_text_success(self):
        r = CommandResult(True, "echo hello", stdout="hello world")
        assert r.to_text() == "hello world"

    def test_to_text_failure(self):
        r = CommandResult(False, "bad", stderr="something failed", exit_code=1)
        assert "something failed" in r.to_text()

    def test_to_text_timeout(self):
        r = CommandResult(False, "slow", execution_time=30.0, timed_out=True)
        assert "timed out" in r.to_text()

    def test_to_text_no_stdout(self):
        r = CommandResult(True, "cmd")
        assert "successfully" in r.to_text()


class TestCLIExecutor:
    def test_init(self):
        executor = CLIExecutor()
        assert executor._shell == ShellType.POWERSHELL

    def test_validate_empty(self):
        executor = CLIExecutor()
        assert executor._validate_command("") is False
        assert executor._validate_command(None) is False

    def test_validate_blocked(self):
        executor = CLIExecutor(blocked_commands=["rm", "format"])
        assert executor._validate_command("rm -rf /") is False
        assert executor._validate_command("format c:") is False
        assert executor._validate_command("echo hello") is True

    def test_validate_dangerous(self):
        executor = CLIExecutor()
        assert executor._validate_command("format c:") is False
        assert executor._validate_command("shutdown /s") is False
        assert executor._validate_command("del /f /q") is False

    def test_validate_semicolon(self):
        executor = CLIExecutor()
        assert executor._validate_command("echo a; echo b") is False

    def test_validate_dollar_paren(self):
        executor = CLIExecutor()
        assert executor._validate_command("echo $(whoami)") is False

    def test_validate_good(self):
        executor = CLIExecutor()
        assert executor._validate_command("echo hello") is True
        assert executor._validate_command("dir") is True

    def test_execute_echo(self):
        executor = CLIExecutor(shell=ShellType.POWERSHELL)
        r = executor.execute("Write-Output 'hello'")
        assert r.success is True
        assert "hello" in r.stdout

    def test_execute_failure(self):
        executor = CLIExecutor(shell=ShellType.POWERSHELL)
        r = executor.execute("Write-Error 'test error'")
        assert r.success is False

    def test_execute_blocked(self):
        executor = CLIExecutor(blocked_commands=["rm"])
        r = executor.execute("rm -rf /")
        assert r.success is False
        assert "blocked" in r.stderr.lower()

    def test_execute_timeout(self):
        executor = CLIExecutor(shell=ShellType.POWERSHELL)
        r = executor.execute("Start-Sleep -Seconds 10", timeout=0.5)
        assert r.success is False
        assert r.timed_out is True

    def test_execute_safe(self):
        executor = CLIExecutor(shell=ShellType.POWERSHELL)
        r = executor.execute_safe("Write-Output 'safe'")
        assert r.success is True

    def test_build_environment(self):
        executor = CLIExecutor(env_passthrough=True)
        env = executor._build_environment({"CUSTOM": "value"})
        assert "CUSTOM" in env
        assert "PATH" in env

    def test_build_environment_restricted(self):
        executor = CLIExecutor(
            env_passthrough=False,
            restricted_env={"MY_VAR": "123"},
        )
        env = executor._build_environment()
        assert "MY_VAR" in env
        assert "HOME" not in env

    def test_get_shell_info(self):
        executor = CLIExecutor()
        info = executor.get_shell_info()
        assert "shell" in info
        assert "workspace" in info

    def test_execute_with_working_dir(self, tmp_path):
        executor = CLIExecutor(shell=ShellType.POWERSHELL)
        r = executor.execute("Get-Location", working_dir=str(tmp_path))
        assert r.success is True


class TestCLIExecutorTool:
    def test_init(self):
        tool = CLIExecutorTool()
        assert tool._executor is not None

    def test_execute_command(self):
        tool = CLIExecutorTool()
        r = tool.execute_command("Write-Output 'test'")
        assert r.success is True

    def test_execute_safe(self):
        tool = CLIExecutorTool()
        r = tool.execute_safe("Write-Output 'safe'")
        assert r.success is True

    def test_get_info(self):
        tool = CLIExecutorTool()
        info = tool.get_info()
        assert "shell" in info

    def test_execute_with_custom_executor(self):
        executor = CLIExecutor(blocked_commands=["rm"])
        tool = CLIExecutorTool(executor)
        r = tool.execute_command("rm -rf /")
        assert r.success is False
