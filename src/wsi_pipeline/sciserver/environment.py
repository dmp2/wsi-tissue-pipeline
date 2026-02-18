"""
SciServer Environment Detection

Utilities for detecting whether code is running in SciServer
Compute environment and retrieving user information.
"""

from __future__ import annotations
from typing import Optional


def is_sciserver_environment() -> bool:
    """
    Check if running in SciServer Compute environment.
    
    Returns
    -------
    bool
        True if running in SciServer Compute, False otherwise.
    """
    try:
        from SciServer import Config
        return Config.isSciServerComputeEnvironment()
    except ImportError:
        return False
    except Exception:
        return False


def get_sciserver_user() -> Optional[str]:
    """
    Get current SciServer username.
    
    Returns
    -------
    str or None
        Username if in SciServer environment, None otherwise.
    """
    if not is_sciserver_environment():
        return None
    try:
        from SciServer import Authentication
        user = Authentication.getKeystoneUserWithToken()
        return user.userName if user else None
    except Exception:
        return None


def get_sciserver_token() -> Optional[str]:
    """
    Get current SciServer authentication token.
    
    Returns
    -------
    str or None
        Token if available, None otherwise.
    """
    if not is_sciserver_environment():
        return None
    try:
        from SciServer import Authentication
        return Authentication.getToken()
    except Exception:
        return None


__all__ = [
    "is_sciserver_environment",
    "get_sciserver_user",
    "get_sciserver_token",
]
