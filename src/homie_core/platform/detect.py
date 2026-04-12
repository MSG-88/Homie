import sys

from homie_core.platform.base import PlatformAdapter


def get_platform_adapter() -> PlatformAdapter:
    if sys.platform == "win32":
        from homie_core.platform.windows import WindowsAdapter
        return WindowsAdapter()
    elif sys.platform == "darwin":
        from homie_core.platform.macos import MacOSAdapter
        return MacOSAdapter()
    else:
        from homie_core.platform.linux import LinuxAdapter
        return LinuxAdapter()
