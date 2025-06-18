import unittest
from utils.file_operations import SafeFileOps
from utils.resource_monitor import ResourceMonitor
from pathlib import Path


class TestUtils(unittest.TestCase):
    def test_file_reading(self):
        # non-existent file returns None
        self.assertIsNone(SafeFileOps.read_file_safely("nonexistent.txt"))

        # valid file returns content
        with SafeFileOps.atomic_write(Path("test.txt")) as f:
            f.write("test content")
        self.assertIsNotNone(SafeFileOps.read_file_safely("test.txt"))

    def test_memory_monitoring(self):
        self.assertTrue(ResourceMonitor.check_memory_usage())


if __name__ == "__main__":
    unittest.main()
