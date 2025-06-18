import unittest
from utils.file_operations import read_file_safely
from utils.resource_monitor import ResourceMonitor


class TestUtils(unittest.TestCase):
    def test_file_reading(self):
        # non-existent file returns None
        self.assertIsNone(read_file_safely("nonexistent.txt"))

        # valid file returns content
        with open("test.txt", "w") as f:
            f.write("test content")
        self.assertIsNotNone(read_file_safely("test.txt"))

    def test_memory_monitoring(self):
        stats = ResourceMonitor.get_memory_stats()
        self.assertIn("percent_used", stats)
        self.assertIn("total_gb", stats)


if __name__ == "__main__":
    unittest.main()
