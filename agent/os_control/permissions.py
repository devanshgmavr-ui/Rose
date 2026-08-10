"""OS control permission constants and defaults."""

from ..tools.base import ConfirmationLevel


OS_PERMISSIONS = {
    "os.screen_capture": ConfirmationLevel.ALLOW,
    "os.system_info": ConfirmationLevel.ALLOW,
    "os.mouse": ConfirmationLevel.REQUIRE_CONFIRMATION,
    "os.keyboard": ConfirmationLevel.REQUIRE_CONFIRMATION,
    "os.window": ConfirmationLevel.DENY,
}

OS_PERMISSION_SCOPES = {
    "os.screen_capture": {"*"},
    "os.system_info": {"*"},
    "os.mouse": {"*"},
    "os.keyboard": {"*"},
    "os.window": set(),
}


def register_os_permissions(permission_manager, mouse_enabled: bool = False, keyboard_enabled: bool = False) -> None:
    for perm, scopes in OS_PERMISSION_SCOPES.items():
        if perm == "os.mouse" and not mouse_enabled:
            continue
        if perm == "os.keyboard" and not keyboard_enabled:
            continue
        for scope in scopes:
            permission_manager.grant_permission(perm, scope)
        level = OS_PERMISSIONS.get(perm, ConfirmationLevel.REQUIRE_CONFIRMATION)
        permission_manager.set_confirmation_level(perm, level)
