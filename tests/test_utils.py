import unittest
from pathlib import Path

from utils.file_operations import SafeFileOps, read_file_safely
from utils.resource_monitor import ResourceMonitor
from utils.errors import ErrorAggregator, ValidationError
from config import AppConfig


class TestUtils(unittest.TestCase):
    def test_file_reading(self):
        # non-existent file returns None
        self.assertIsNone(read_file_safely("nonexistent.txt"))

        # valid file returns content
        with open("test.txt", "w") as f:
            f.write("test content")
        self.assertIsNotNone(read_file_safely("test.txt"))

    def test_atomic_write(self):
        path = Path("atomic.txt")
        SafeFileOps.atomic_write(path, "hello")
        self.assertTrue(path.exists())
        self.assertEqual(path.read_text(), "hello")
        path.unlink()

    def test_memory_monitoring(self):
        stats = ResourceMonitor.get_memory_stats()
        self.assertIn("percent_used", stats)
        self.assertIn("total_gb", stats)
        self.assertTrue(ResourceMonitor.check_system_resources())

    def test_error_aggregation_and_config(self):
        aggregator = ErrorAggregator()
        err = ValidationError("bad", {"field": "value"})
        aggregator.add_error(err)
        self.assertEqual(len(aggregator.errors), 1)

        cfg = AppConfig.from_env()
        self.assertTrue(cfg.validate())


if __name__ == "__main__":
    unittest.main()
