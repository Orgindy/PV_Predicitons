from pathlib import Path
from utils.resource_monitor import ResourceCleanup


def test_cleanup_context(tmp_path):
    tmp_file = tmp_path / "temp.tmp"
    tmp_file.write_text("data")
    with ResourceCleanup.cleanup_context(tmp_path):
        pass
    assert not any(tmp_path.glob("*.tmp"))
