"""Permission system for tool execution."""

from typing import Dict, Set, Optional
from .base import Permission, ConfirmationLevel


class PermissionManager:
    """Manages tool permissions and confirmation requirements."""

    def __init__(self):
        self._granted: Dict[str, Set[str]] = {}
        self._confirmation_overrides: Dict[str, ConfirmationLevel] = {}
        self._initialize_defaults()

    def _initialize_defaults(self):
        self._granted = {
            "filesystem.read": {"workspace_only"},
            "filesystem.write": {"workspace_only"},
            "code.execute": {"sandbox_only"},
            "command.execute": set(),
        }
        self._confirmation_overrides = {
            "filesystem.read": ConfirmationLevel.ALLOW,
            "filesystem.write": ConfirmationLevel.REQUIRE_CONFIRMATION,
            "code.execute": ConfirmationLevel.REQUIRE_CONFIRMATION,
            "command.execute": ConfirmationLevel.DENY,
        }

    def has_permission(self, permission: str, context: Optional[str] = None) -> bool:
        perms = self._granted.get(permission, set())
        if not perms:
            return False
        if "workspace_only" in perms and context == "workspace":
            return True
        if "sandbox_only" in perms and context == "sandbox":
            return True
        if "*" in perms:
            return True
        return False

    def get_confirmation_level(self, permission: str) -> ConfirmationLevel:
        return self._confirmation_overrides.get(
            permission, ConfirmationLevel.REQUIRE_CONFIRMATION
        )

    def grant_permission(self, permission: str, scope: str = "*"):
        if permission not in self._granted:
            self._granted[permission] = set()
        self._granted[permission].add(scope)

    def revoke_permission(self, permission: str, scope: Optional[str] = None):
        if scope is None:
            self._granted.pop(permission, None)
        elif permission in self._granted:
            self._granted[permission].discard(scope)

    def set_confirmation_level(self, permission: str, level: ConfirmationLevel):
        self._confirmation_overrides[permission] = level

    def check_tool_permissions(
        self, required_permissions: list, context: Optional[str] = None
    ) -> tuple:
        denied = []
        needs_confirmation = []
        for perm in required_permissions:
            perm_value = perm.value if isinstance(perm, Permission) else perm
            if not self.has_permission(perm_value, context):
                denied.append(perm_value)
            elif self.get_confirmation_level(perm_value) == ConfirmationLevel.REQUIRE_CONFIRMATION:
                needs_confirmation.append(perm_value)
        return len(denied) == 0, denied, needs_confirmation

    def get_all_permissions(self) -> Dict[str, list]:
        return {
            perm: list(scopes)
            for perm, scopes in self._granted.items()
        }
