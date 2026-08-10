"""Vision permission constants and defaults.

Stage 3.1 - Vision Analysis.

Defines vision-specific permissions following the existing
permission architecture. Vision is disabled by default
and requires explicit configuration.
"""

from ..tools.base import ConfirmationLevel


VISION_PERMISSIONS = {
    "vision.analyze": ConfirmationLevel.REQUIRE_CONFIRMATION,
}

VISION_PERMISSION_SCOPES = {
    "vision.analyze": {"*"},
}


def register_vision_permissions(
    permission_manager,
    vision_enabled: bool = False,
) -> None:
    """Register vision permissions with the permission manager.

    Args:
        permission_manager: The PermissionManager instance.
        vision_enabled: Whether vision analysis is enabled.
    """
    if not vision_enabled:
        return

    for perm, scopes in VISION_PERMISSION_SCOPES.items():
        for scope in scopes:
            permission_manager.grant_permission(perm, scope)
        level = VISION_PERMISSIONS.get(
            perm, ConfirmationLevel.REQUIRE_CONFIRMATION
        )
        permission_manager.set_confirmation_level(perm, level)
