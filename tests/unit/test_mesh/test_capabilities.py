import sys
from homie_core.mesh.capabilities import NodeCapabilities, detect_capabilities

def test_detect_returns_capabilities():
    caps = detect_capabilities()
    assert caps.cpu_cores > 0
    assert caps.ram_gb > 0
    assert caps.os in ("windows", "linux", "macos")
    assert isinstance(caps.has_mic, bool)
    assert isinstance(caps.has_display, bool)

def test_capabilities_score_no_gpu():
    caps = NodeCapabilities(
        gpu=None, cpu_cores=8, ram_gb=16.0, disk_free_gb=100.0,
        os="linux", has_mic=True, has_display=True,
        has_model_loaded=False, model_name=None,
    )
    score = caps.capability_score()
    assert score == 40.0

def test_capabilities_score_with_gpu():
    caps = NodeCapabilities(
        gpu={"name": "RTX 5080", "vram_gb": 16.0},
        cpu_cores=16, ram_gb=32.0, disk_free_gb=200.0,
        os="windows", has_mic=True, has_display=True,
        has_model_loaded=True, model_name="Qwen3.5",
    )
    score = caps.capability_score()
    assert score == 290.0

def test_capabilities_to_dict():
    caps = NodeCapabilities(
        gpu=None, cpu_cores=4, ram_gb=8.0, disk_free_gb=50.0,
        os="linux", has_mic=False, has_display=True,
        has_model_loaded=False, model_name=None,
    )
    d = caps.to_dict()
    assert d["cpu_cores"] == 4
    assert d["ram_gb"] == 8.0
    assert d["gpu"] is None

def test_detect_os_platform():
    caps = detect_capabilities()
    if sys.platform == "win32":
        assert caps.os == "windows"
    elif sys.platform == "darwin":
        assert caps.os == "macos"
    else:
        assert caps.os == "linux"
