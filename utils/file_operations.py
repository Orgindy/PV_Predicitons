"""Safe file operation utilities."""

import logging
import os
import time
from pathlib import Path
from typing import Optional, Union


MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10 MB


class SafeFileOps:
    """Provide safer primitives for file writing."""

    @staticmethod
    def atomic_write(path: Path, content: str):
        """Write *content* to *path* atomically."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with tmp.open("w") as f:
                f.write(content)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


class FileLock:
    """Simple file lock using lock files."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock_file = self.path.with_suffix(self.path.suffix + ".lock")

    def _acquire(self):
        while True:
            try:
                self.lock_fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                break
            except FileExistsError:
                time.sleep(0.1)

    def _release(self):
        try:
            os.close(self.lock_fd)
        except Exception:
            pass
        try:
            os.remove(self.lock_file)
        except FileNotFoundError:
            pass

    def __enter__(self):
        self._acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._release()


def read_file_safely(path: Union[str, Path], encoding: str = "utf-8") -> Optional[str]:
    """Return file contents if within size limit and accessible."""
    path = Path(path)
    try:
        if not path.exists():
            logging.warning(f"File not found: {path}")
            return None
        if path.stat().st_size > MAX_FILE_SIZE:
            logging.warning(f"File {path} exceeds max size of {MAX_FILE_SIZE} bytes")
            return None
        with path.open("r", encoding=encoding) as f:
            return f.read()
    except Exception as exc:
        logging.error(f"Failed to read {path}: {exc}")
        return None
