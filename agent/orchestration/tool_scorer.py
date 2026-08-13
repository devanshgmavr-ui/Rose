"""Tool scorer for dynamic tool selection.

Phase 9 - Dynamic Tool Selection with Scoring.

Scores candidate tools based on capability match, permissions,
risk, reliability, and context. Selects the best tool for each step.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from .tool_catalog import ToolMetadata, build_tool_catalog
from .capability_analyzer import Capability, CapabilityAnalysis

logger = logging.getLogger(__name__)


# Mapping from capabilities to tool categories/names
CAPABILITY_TOOLS_MAP: Dict[str, List[str]] = {
    "file_operations": ["filesystem"],
    "code_execution": ["python_sandbox"],
    "screen_capture": ["screen_capture"],
    "system_information": ["system_info"],
    "mouse_control": ["mouse"],
    "keyboard_input": ["keyboard"],
    "window_management": ["window"],
    "browser_automation": ["browser"],
    "browser_reading": ["browser"],
    "browser_interaction": ["browser"],
    "vision_analysis": ["vision_analyze", "image_analyze"],
    "visual_grounding": ["visual_ground"],
    "app_launch": ["launch_app"],
    "text_transcription": ["vision_analyze", "keyboard"],
    "verification": ["screen_capture", "vision_analyze"],
}


@dataclass
class ToolScore:
    """Score for a candidate tool."""
    tool_name: str
    total_score: float
    capability_match: float
    permission_score: float
    risk_score: float
    reliability_score: float
    context_score: float
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "total_score": round(self.total_score, 3),
            "capability_match": round(self.capability_match, 3),
            "permission_score": round(self.permission_score, 3),
            "risk_score": round(self.risk_score, 3),
            "reliability_score": round(self.reliability_score, 3),
            "context_score": round(self.context_score, 3),
            "reason": self.reason,
        }


@dataclass
class SelectionResult:
    """Result of tool selection for a step."""
    selected_tool: Optional[ToolScore]
    alternatives: List[ToolScore]
    all_candidates: List[ToolScore]
    reason_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_tool": self.selected_tool.to_dict() if self.selected_tool else None,
            "alternatives": [a.to_dict() for a in self.alternatives],
            "candidate_count": len(self.all_candidates),
            "reason_summary": self.reason_summary,
        }


class ToolScorer:
    """Scores and selects the best tool for each step.

    Uses capability matching, permission checking, risk assessment,
    and context awareness to select optimal tools.
    """

    def __init__(self, tool_registry=None, permission_manager=None):
        self._catalog = build_tool_catalog()
        self._tool_registry = tool_registry
        self._permission_manager = permission_manager
        self._failure_counts: Dict[str, int] = {}
        self._success_counts: Dict[str, int] = {}

    def record_failure(self, tool_name: str):
        """Record a tool failure for reliability scoring."""
        self._failure_counts[tool_name] = self._failure_counts.get(tool_name, 0) + 1

    def record_success(self, tool_name: str):
        """Record a tool success for reliability scoring."""
        self._success_counts[tool_name] = self._success_counts.get(tool_name, 0) + 1

    def reset_history(self):
        """Reset failure/success history."""
        self._failure_counts.clear()
        self._success_counts.clear()

    def select_tool(
        self,
        capabilities: List[Capability],
        context: Optional[Dict[str, Any]] = None,
        explicit_tools: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> SelectionResult:
        """Select the best tool for given capabilities.

        Args:
            capabilities: Required capabilities for this step.
            context: Current execution context (active tools, previous results, etc.)
            explicit_tools: Tools explicitly requested by user.
            constraints: User constraints (prohibited tools, etc.)

        Returns:
            SelectionResult with best tool and alternatives.
        """
        context = context or {}
        constraints = constraints or {}
        prohibited = set(constraints.get("prohibited_tools", []))

        # Get candidate tools from capabilities
        candidate_names = self._get_candidate_tools(capabilities, explicit_tools)

        # Filter prohibited
        candidate_names = [n for n in candidate_names if n not in prohibited]

        # Score each candidate
        scored = []
        for tool_name in candidate_names:
            if tool_name not in self._catalog:
                continue
            meta = self._catalog[tool_name]
            score = self._score_tool(meta, capabilities, context, constraints)
            scored.append(score)

        scored.sort(key=lambda s: s.total_score, reverse=True)

        selected = scored[0] if scored else None
        alternatives = scored[1:3] if len(scored) > 1 else []

        reason = ""
        if selected:
            reason = self._generate_reason(selected, capabilities)
        elif not candidate_names:
            reason = "No tools match the required capabilities"
        else:
            reason = "All candidate tools were filtered by constraints or permissions"

        return SelectionResult(
            selected_tool=selected,
            alternatives=alternatives,
            all_candidates=scored,
            reason_summary=reason,
        )

    def select_tool_for_step(
        self,
        step_description: str,
        step_action: str = "",
        step_arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        explicit_tools: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> SelectionResult:
        """Select the best tool for a specific plan step.

        Convenience method that combines capability analysis with selection.
        """
        from .capability_analyzer import CapabilityAnalyzer
        analyzer = CapabilityAnalyzer()
        capabilities = analyzer.get_capabilities_for_step(step_description, step_action)

        if not capabilities:
            # Fallback: try to match step tool_name from catalog
            if step_arguments and "tool_name" in step_arguments:
                tool_name = step_arguments["tool_name"]
                if tool_name in self._catalog:
                    capabilities = [Capability(
                        name=f"direct_{tool_name}",
                        description=f"Direct tool request: {tool_name}",
                        confidence=1.0,
                    )]

        return self.select_tool(
            capabilities=capabilities,
            context=context,
            explicit_tools=explicit_tools,
            constraints=constraints,
        )

    def _get_candidate_tools(
        self,
        capabilities: List[Capability],
        explicit_tools: Optional[List[str]] = None,
    ) -> List[str]:
        """Get candidate tool names from capabilities."""
        candidates = set()

        # If explicit tools are specified, prioritize them
        if explicit_tools:
            for tool in explicit_tools:
                if tool in self._catalog:
                    candidates.add(tool)

        # Map capabilities to tools
        for cap in capabilities:
            if cap.name in CAPABILITY_TOOLS_MAP:
                for tool in CAPABILITY_TOOLS_MAP[cap.name]:
                    candidates.add(tool)

        # If no candidates found, return all catalog tools as fallback
        if not candidates:
            candidates = set(self._catalog.keys())

        return list(candidates)

    def _score_tool(
        self,
        meta: ToolMetadata,
        capabilities: List[Capability],
        context: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> ToolScore:
        """Score a single tool against capabilities and context."""
        cap_match = self._score_capability_match(meta, capabilities)
        perm_score = self._score_permissions(meta)
        risk_score = self._score_risk(meta)
        rel_score = self._score_reliability(meta.name)
        ctx_score = self._score_context(meta, context)

        # Weighted total
        total = (
            cap_match * 0.35
            + perm_score * 0.25
            + rel_score * 0.20
            + ctx_score * 0.15
            + risk_score * 0.05
        )

        return ToolScore(
            tool_name=meta.name,
            total_score=total,
            capability_match=cap_match,
            permission_score=perm_score,
            risk_score=risk_score,
            reliability_score=rel_score,
            context_score=ctx_score,
        )

    def _score_capability_match(
        self,
        meta: ToolMetadata,
        capabilities: List[Capability],
    ) -> float:
        """Score how well the tool matches required capabilities."""
        if not capabilities:
            return 0.5

        tool_capabilities = set()
        for cap in capabilities:
            if cap.name in CAPABILITY_TOOLS_MAP:
                for tool in CAPABILITY_TOOLS_MAP[cap.name]:
                    tool_capabilities.add(tool)

        if meta.name in tool_capabilities:
            # Calculate average confidence of matching capabilities
            matching_confs = []
            for cap in capabilities:
                if cap.name in CAPABILITY_TOOLS_MAP:
                    if meta.name in CAPABILITY_TOOLS_MAP[cap.name]:
                        matching_confs.append(cap.confidence)
            return sum(matching_confs) / len(matching_confs) if matching_confs else 0.8

        # Check if tool is in same category
        for cap in capabilities:
            if cap.name in CAPABILITY_TOOLS_MAP:
                if meta.category in ("os_control", "browser", "vision", "filesystem", "code"):
                    return 0.3

        return 0.1

    def _score_permissions(self, meta: ToolMetadata) -> float:
        """Score based on permission requirements."""
        if not meta.permissions:
            return 1.0

        if not self._permission_manager:
            return 0.7

        allowed_count = 0
        for perm in meta.permissions:
            status = self._permission_manager.check_permission(perm)
            if status.value == "allow":
                allowed_count += 1
            elif status.value == "require_confirmation":
                allowed_count += 0.5

        return allowed_count / len(meta.permissions) if meta.permissions else 1.0

    def _score_risk(self, meta: ToolMetadata) -> float:
        """Score based on risk level (higher = worse)."""
        failure_count = len(meta.failure_modes)
        if failure_count == 0:
            return 1.0
        elif failure_count <= 2:
            return 0.8
        elif failure_count <= 4:
            return 0.6
        return 0.4

    def _score_reliability(self, tool_name: str) -> float:
        """Score based on historical success/failure rate."""
        successes = self._success_counts.get(tool_name, 0)
        failures = self._failure_counts.get(tool_name, 0)
        total = successes + failures

        if total == 0:
            return 0.7  # Unknown reliability

        return successes / total

    def _score_context(self, meta: ToolMetadata, context: Dict[str, Any]) -> float:
        """Score based on current execution context."""
        score = 0.5

        # Bonus if tool is already active/initialized
        active_tools = context.get("active_tools", [])
        if meta.name in active_tools:
            score += 0.3

        # Bonus if tool recently succeeded
        recent_successes = context.get("recent_successes", [])
        if meta.name in recent_successes:
            score += 0.2

        # Penalty if tool recently failed
        recent_failures = context.get("recent_failures", [])
        if meta.name in recent_failures:
            score -= 0.3

        return max(0.0, min(1.0, score))

    def _generate_reason(
        self,
        score: ToolScore,
        capabilities: List[Capability],
    ) -> str:
        """Generate a short reason for the selection."""
        cap_names = [c.name for c in capabilities[:2]]
        if score.capability_match > 0.7:
            return f"Best match for {', '.join(cap_names)} capability"
        elif score.permission_score > 0.8:
            return f"Available and permitted for {', '.join(cap_names)}"
        elif score.reliability_score > 0.8:
            return f"Highly reliable for {', '.join(cap_names)}"
        return f"Selected for {', '.join(cap_names)} based on overall score"
