"""Browser permission constants and defaults.

Stage 2.4.5 - Browser screenshot support.

Defines browser-specific permissions following the existing
OS permission architecture. Browser automation is disabled
by default and requires explicit configuration.
"""

from ..tools.base import ConfirmationLevel


BROWSER_PERMISSIONS = {
    "browser.session": ConfirmationLevel.REQUIRE_CONFIRMATION,
    "browser.navigation": ConfirmationLevel.REQUIRE_CONFIRMATION,
    "browser.page_read": ConfirmationLevel.REQUIRE_CONFIRMATION,
    "browser.inspect": ConfirmationLevel.ALLOW,
    "browser.interact": ConfirmationLevel.REQUIRE_CONFIRMATION,
    "browser.screenshot": ConfirmationLevel.REQUIRE_CONFIRMATION,
}

BROWSER_PERMISSION_SCOPES = {
    "browser.session": {"*"},
    "browser.navigation": {"*"},
    "browser.page_read": {"*"},
    "browser.inspect": {"*"},
    "browser.interact": {"*"},
    "browser.screenshot": {"*"},
}


def register_browser_permissions(
    permission_manager,
    browser_enabled: bool = False,
) -> None:
    """Register browser permissions with the permission manager.

    Args:
        permission_manager: The PermissionManager instance.
        browser_enabled: Whether browser automation is enabled.
    """
    if not browser_enabled:
        return

    for perm, scopes in BROWSER_PERMISSION_SCOPES.items():
        for scope in scopes:
            permission_manager.grant_permission(perm, scope)
        level = BROWSER_PERMISSIONS.get(
            perm, ConfirmationLevel.REQUIRE_CONFIRMATION
        )
        permission_manager.set_confirmation_level(perm, level)
