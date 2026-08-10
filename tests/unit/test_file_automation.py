"""Tests for Stage 5.2 - Advanced File Automation."""

import pytest
import os
import tempfile
from agent.tools.file_automation import FileAutomator, FileResult, FileInfo, FileOperation


class TestFileResult:
    def test_success(self):
        r = FileResult(True, "read", "/test.txt", output="hello")
        assert r.success is True
        assert r.output == "hello"

    def test_failure(self):
        r = FileResult(False, "read", "/test.txt", error="not found")
        assert r.success is False
        assert r.error == "not found"

    def test_to_dict(self):
        r = FileResult(True, "write", "/test.txt", bytes_written=100)
        d = r.to_dict()
        assert d["success"] is True
        assert d["bytes_written"] == 100

    def test_to_text_success_with_output(self):
        r = FileResult(True, "read", "/test.txt", output="file contents")
        assert r.to_text() == "file contents"

    def test_to_text_success_no_output(self):
        r = FileResult(True, "write", "/test.txt")
        assert "completed" in r.to_text()

    def test_to_text_failure(self):
        r = FileResult(False, "read", "/test.txt", error="not found")
        assert "not found" in r.to_text()


class TestFileInfo:
    def test_creation(self):
        info = FileInfo(name="test.txt", path="/test.txt", size=100, modified=1.0, is_dir=False)
        assert info.name == "test.txt"
        assert info.size == 100

    def test_to_dict(self):
        info = FileInfo(name="test.txt", path="/test.txt", size=100, modified=1.0, is_dir=False)
        d = info.to_dict()
        assert d["name"] == "test.txt"
        assert d["is_dir"] is False


class TestFileAutomator:
    def test_init(self):
        automator = FileAutomator()
        assert automator._workspace

    def test_read_file_not_found(self):
        automator = FileAutomator()
        r = automator.read_file("nonexistent_file_xyz.txt")
        assert r.success is False

    def test_read_file(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world", encoding="utf-8")
        automator = FileAutomator(workspace_dir=str(tmp_path))
        r = automator.read_file("test.txt")
        assert r.success is True
        assert r.output == "hello world"

    def test_write_file(self, tmp_path):
        automator = FileAutomator(workspace_dir=str(tmp_path))
        r = automator.write_file("output.txt", "test content")
        assert r.success is True
        assert r.bytes_written > 0
        assert (tmp_path / "output.txt").exists()

    def test_write_file_creates_dirs(self, tmp_path):
        automator = FileAutomator(workspace_dir=str(tmp_path))
        r = automator.write_file("sub/dir/file.txt", "nested content")
        assert r.success is True
        assert (tmp_path / "sub" / "dir" / "file.txt").exists()

    def test_append_file(self, tmp_path):
        automator = FileAutomator(workspace_dir=str(tmp_path))
        automator.write_file("test.txt", "hello")
        r = automator.write_file("test.txt", " world", append=True)
        assert r.success is True
        content = automator.read_file("test.txt")
        assert content.output == "hello world"

    def test_copy_file(self, tmp_path):
        automator = FileAutomator(workspace_dir=str(tmp_path))
        automator.write_file("src.txt", "content")
        r = automator.copy_file("src.txt", "dst.txt")
        assert r.success is True
        assert (tmp_path / "dst.txt").exists()

    def test_copy_file_no_overwrite(self, tmp_path):
        automator = FileAutomator(workspace_dir=str(tmp_path))
        automator.write_file("src.txt", "content")
        automator.write_file("dst.txt", "existing")
        r = automator.copy_file("src.txt", "dst.txt", overwrite=False)
        assert r.success is False

    def test_move_file(self, tmp_path):
        automator = FileAutomator(workspace_dir=str(tmp_path))
        automator.write_file("src.txt", "content")
        r = automator.move_file("src.txt", "dst.txt")
        assert r.success is True
        assert not (tmp_path / "src.txt").exists()
        assert (tmp_path / "dst.txt").exists()

    def test_delete_file(self, tmp_path):
        automator = FileAutomator(workspace_dir=str(tmp_path))
        automator.write_file("to_delete.txt", "bye")
        r = automator.delete_file("to_delete.txt")
        assert r.success is True
        assert not (tmp_path / "to_delete.txt").exists()

    def test_delete_file_not_found(self):
        automator = FileAutomator()
        r = automator.delete_file("nonexistent.txt")
        assert r.success is False

    def test_list_directory(self, tmp_path):
        automator = FileAutomator(workspace_dir=str(tmp_path))
        automator.write_file("a.txt", "a")
        automator.write_file("b.txt", "b")
        (tmp_path / "subdir").mkdir()
        r = automator.list_directory(".")
        assert r.success is True
        assert r.files_affected >= 3

    def test_list_directory_recursive(self, tmp_path):
        automator = FileAutomator(workspace_dir=str(tmp_path))
        automator.write_file("top.txt", "top")
        (tmp_path / "sub").mkdir()
        automator.write_file("sub/bottom.txt", "bottom")
        r = automator.list_directory(".", recursive=True)
        assert r.success is True
        assert r.files_affected >= 2

    def test_make_directory(self, tmp_path):
        automator = FileAutomator(workspace_dir=str(tmp_path))
        r = automator.make_directory("new_dir")
        assert r.success is True
        assert (tmp_path / "new_dir").is_dir()

    def test_get_file_info(self, tmp_path):
        automator = FileAutomator(workspace_dir=str(tmp_path))
        automator.write_file("info.txt", "content")
        r = automator.get_file_info("info.txt")
        assert r.success is True
        assert r.output["name"] == "info.txt"

    def test_search_in_file(self, tmp_path):
        automator = FileAutomator(workspace_dir=str(tmp_path))
        automator.write_file("search.txt", "line1\nhello world\nline3\nhello again")
        r = automator.search_in_file("search.txt", "hello")
        assert r.success is True
        assert r.files_affected == 2

    def test_replace_in_file(self, tmp_path):
        automator = FileAutomator(workspace_dir=str(tmp_path))
        automator.write_file("replace.txt", "hello world")
        r = automator.replace_in_file("replace.txt", "world", "python")
        assert r.success is True
        assert automator.read_file("replace.txt").output == "hello python"

    def test_replace_not_found(self, tmp_path):
        automator = FileAutomator(workspace_dir=str(tmp_path))
        automator.write_file("replace.txt", "hello world")
        r = automator.replace_in_file("replace.txt", "xyz", "abc")
        assert r.success is False

    def test_get_checksum(self, tmp_path):
        automator = FileAutomator(workspace_dir=str(tmp_path))
        automator.write_file("checksum.txt", "test data")
        r = automator.get_checksum("checksum.txt")
        assert r.success is True
        assert len(r.output) == 64

    def test_read_file_too_large(self, tmp_path):
        automator = FileAutomator(workspace_dir=str(tmp_path), max_file_size_mb=0.001)
        automator.write_file("big.txt", "x" * 2000)
        r = automator.read_file("big.txt")
        assert r.success is False

    def test_list_directory_not_found(self):
        automator = FileAutomator()
        r = automator.list_directory("nonexistent_dir_xyz")
        assert r.success is False

    def test_get_file_info_not_found(self):
        automator = FileAutomator()
        r = automator.get_file_info("nonexistent.txt")
        assert r.success is False

    def test_copy_source_not_found(self, tmp_path):
        automator = FileAutomator(workspace_dir=str(tmp_path))
        r = automator.copy_file("nope.txt", "dst.txt")
        assert r.success is False

    def test_move_source_not_found(self, tmp_path):
        automator = FileAutomator(workspace_dir=str(tmp_path))
        r = automator.move_file("nope.txt", "dst.txt")
        assert r.success is False
