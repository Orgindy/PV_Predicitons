import os
import fcntl
import logging
from pathlib import Path
from typing import Optional, Union, Iterator, IO, Any
from contextlib import contextmanager


class SafeFileOps:
    """Safe file operations with atomic writes and locking."""

    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 100 * 1024 * 1024))

    @staticmethod
    @contextmanager
    def atomic_write(path: Path) -> Iterator[IO[str]]:
        """Context manager for atomic file writes."""
        tmp = path.with_suffix(".tmp")
        try:
            with tmp.open("w") as f:
                yield f
            tmp.replace(path)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    @staticmethod
    def read_file_safely(
        path: Union[str, Path], encoding: str = "utf-8"
    ) -> Optional[str]:
        """Read file with size and encoding checks."""
        path = Path(path)
        try:
            if not path.exists():
                logging.warning(f"File not found: {path}")
                return None
            if path.stat().st_size > SafeFileOps.MAX_FILE_SIZE:
                logging.warning(f"File too large: {path}")
                return None
            with path.open("r", encoding=encoding) as f:
                return f.read()
        except Exception as e:
            logging.error(f"Failed to read {path}: {e}")
            return None


class FileLock:
    """File-based locking mechanism."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock_file = path.with_suffix(".lock")
        self._fd: Optional[IO[str]] = None

    def __enter__(self) -> "FileLock":
        self._acquire()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        self._release()

    def _acquire(self) -> None:
        """Acquire a file lock."""
        self._fd = open(self.lock_file, "w")
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError:
            self._fd.close()
            raise IOError(f"Failed to acquire lock on {self.lock_file}")

    def _release(self) -> None:
        """Release the file lock."""
        if self._fd:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            self._fd.close()
            try:
                self.lock_file.unlink()
            except Exception:
                pass
