"""Tool router for executing validated tool requests."""

import time
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from .base import Tool, ToolRequest, ToolResult, ConfirmationLevel
from .registry import ToolRegistry
from .permissions import PermissionManager
from .audit import AuditLogger, AuditRecord


class ToolRouter:
    """Routes and executes tool requests with validation and safety checks."""

    def __init__(
        self,
        registry: ToolRegistry,
        permission_manager: Optional[PermissionManager] = None,
        audit_logger: Optional[AuditLogger] = None,
        default_timeout: float = 30.0,
        max_output_size: int = 10000,
        max_tool_calls_per_request: int = 10,
    ):
        self.registry = registry
        self.permissions = permission_manager or PermissionManager()
        self.audit = audit_logger or AuditLogger()
        self.default_timeout = default_timeout
        self.max_output_size = max_output_size
        self.max_tool_calls_per_request = max_tool_calls_per_request
        self._call_count = 0

    def reset_call_count(self):
        self._call_count = 0

    def route(self, request: ToolRequest, context: str = "workspace") -> ToolResult:
        self._call_count += 1
        if self._call_count > self.max_tool_calls_per_request:
            return ToolResult(
                success=False,
                tool_name=request.tool,
                error=f"Maximum tool calls per request exceeded ({self.max_tool_calls_per_request})",
            )

        tool = self.registry.get(request.tool)
        if tool is None:
            return ToolResult(
                success=False,
                tool_name=request.tool,
                error=f"Unknown tool: {request.tool}",
            )

        valid, errors = tool.validate(request.arguments)
        if not valid:
            return ToolResult(
                success=False,
                tool_name=request.tool,
                error=f"Invalid arguments: {'; '.join(errors)}",
            )

        has_perms, denied, needs_confirm = self.permissions.check_tool_permissions(
            tool.required_permissions, context
        )
        if not has_perms:
            record = self.audit.log_request(
                request.tool, request.arguments, request.session_id, "denied"
            )
            self.audit.finalize_record(
                record, False, error=f"Permission denied: {', '.join(denied)}"
            )
            return ToolResult(
                success=False,
                tool_name=request.tool,
                error=f"Permission denied: {', '.join(denied)}",
            )

        perm_decision = "confirmed" if needs_confirm else "auto_approved"
        if needs_confirm and tool.confirmation_level == ConfirmationLevel.DENY:
            perm_decision = "denied"
            record = self.audit.log_request(
                request.tool, request.arguments, request.session_id, perm_decision
            )
            self.audit.finalize_record(record, False, error="Operation requires confirmation but is denied")
            return ToolResult(
                success=False,
                tool_name=request.tool,
                error="Operation requires confirmation but is denied",
            )

        record = self.audit.log_request(
            request.tool, request.arguments, request.session_id, perm_decision
        )

        timeout = tool.timeout or self.default_timeout
        start_time = time.time()

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(tool.execute, request.arguments)
                result = future.result(timeout=timeout)
        except FuturesTimeout:
            elapsed = time.time() - start_time
            self.audit.finalize_record(
                record, False, error="Tool execution timed out", execution_time=elapsed
            )
            return ToolResult(
                success=False,
                tool_name=request.tool,
                error=f"Tool execution timed out after {timeout}s",
                execution_time=elapsed,
            )
        except Exception as e:
            elapsed = time.time() - start_time
            self.audit.finalize_record(
                record, False, error=str(e), execution_time=elapsed
            )
            return ToolResult(
                success=False,
                tool_name=request.tool,
                error=f"Tool execution failed: {str(e)}",
                execution_time=elapsed,
            )

        if result.output and len(result.output) > self.max_output_size:
            result.output = result.output[: self.max_output_size] + "\n...[output truncated]"
            result.truncated = True

        self.audit.finalize_record(
            record,
            result.success,
            result.output,
            result.error,
            result.execution_time,
            result.truncated,
        )

        return result

    def execute_tool(
        self, tool_name: str, arguments: Dict[str, Any], session_id: Optional[str] = None
    ) -> ToolResult:
        request = ToolRequest(tool=tool_name, arguments=arguments, session_id=session_id)
        return self.route(request)
