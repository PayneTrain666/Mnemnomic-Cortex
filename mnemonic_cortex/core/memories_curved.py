"""
Plain-language summary
----------------------
What this file is for: Compatibility import shim for older paths under mnemonic_cortex/core/.
How it fits in the system: Re-exports the real implementation from elsewhere so old imports keep working.
Status: LEGACY / SHIM
Important notes for non-coders: Prefer importing the real module at package root or the named implementation file.
"""

from ..memory_curved import EnhancedCurvedMemory

__all__ = ["EnhancedCurvedMemory"]

