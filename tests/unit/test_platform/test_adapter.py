import sys
from homie_core.platform.detect import get_platform_adapter


def test_correct_adapter_for_platform():
    adapter = get_platform_adapter()
    class_name = type(adapter).__name__
    if sys.platform == "win32":
        assert class_name == "WindowsAdapter"
    elif sys.platform == "darwin":
        assert class_name == "MacOSAdapter"
    else:
        assert class_name == "LinuxAdapter"
