# Minimal global timer singleton holder to avoid importing the core package from model.py
# This module should have no dependencies on the core package to prevent circular imports.
from typing import Any, Optional

_GLOBAL_TIMER: Optional[Any] = None

def set_global_timer(timer: Any) -> None:
    global _GLOBAL_TIMER
    _GLOBAL_TIMER = timer


def get_global_timer() -> Optional[Any]:
    return _GLOBAL_TIMER

