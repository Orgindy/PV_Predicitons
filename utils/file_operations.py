"""Safe file operation utilities."""
import os
import logging
from pathlib import Path
from typing import Optional, Union


MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10 MB


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
