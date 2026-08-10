"""Audit logging for tool executions."""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class AuditRecord:
    timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    permission_decision: str = "pending"
    success: bool = False
    output: str = ""
    error: str = ""
    execution_time: float = 0.0
    truncated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "tool_name": self.tool_name,
            "arguments": self._sanitize_args(self.arguments),
            "permission_decision": self.permission_decision,
            "success": self.success,
            "output": self.output[:500] if self.output else "",
            "error": self.error,
            "execution_time": self.execution_time,
            "truncated": self.truncated,
        }

    def _sanitize_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = {}
        for k, v in args.items():
            if isinstance(v, str) and len(v) > 200:
                sanitized[k] = v[:200] + "...[truncated]"
            else:
                sanitized[k] = v
        return sanitized


class AuditLogger:
    """Structured audit logging for tool executions."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_dir / "tool_audit.jsonl"

    def log_execution(self, record: AuditRecord) -> bool:
        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            return True
        except Exception:
            return False

    def log_request(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        session_id: Optional[str] = None,
        permission_decision: str = "pending",
    ) -> AuditRecord:
        record = AuditRecord(
            session_id=session_id,
            tool_name=tool_name,
            arguments=arguments,
            permission_decision=permission_decision,
        )
        return record

    def finalize_record(
        self,
        record: AuditRecord,
        success: bool,
        output: str = "",
        error: str = "",
        execution_time: float = 0.0,
        truncated: bool = False,
    ):
        record.success = success
        record.output = output
        record.error = error
        record.execution_time = execution_time
        record.truncated = truncated
        self.log_execution(record)

    def get_recent_records(self, limit: int = 50) -> list:
        if not self._log_file.exists():
            return []
        records = []
        try:
            with open(self._log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[-limit:]:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        except Exception:
            pass
        return records

    def get_record_count(self) -> int:
        if not self._log_file.exists():
            return 0
        try:
            with open(self._log_file, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def clear_logs(self):
        if self._log_file.exists():
            self._log_file.unlink()
