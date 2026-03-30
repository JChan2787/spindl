"""Project path utilities for CWD-independent path resolution."""

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """
    Find the spindl project root by walking up from this file
    until we find pyproject.toml.

    Returns:
        Absolute Path to the project root directory.

    Raises:
        RuntimeError: If pyproject.toml cannot be found in any ancestor.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError(
        "Could not find project root (no pyproject.toml in ancestors of "
        f"{Path(__file__).resolve()})"
    )


def resolve_relative_path(path_str: str) -> str:
    """
    If path_str is relative, resolve it against the project root.
    If absolute, return as-is.

    Args:
        path_str: A filesystem path string (absolute or relative).

    Returns:
        Absolute path string.
    """
    p = Path(path_str)
    if p.is_absolute():
        return path_str
    return str(get_project_root() / p)
