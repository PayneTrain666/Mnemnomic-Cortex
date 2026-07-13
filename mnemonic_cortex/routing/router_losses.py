"""
Plain-language summary
----------------------
What this file is for: Compatibility import shim for older paths under mnemonic_cortex/routing/.
How it fits in the system: Re-exports the real implementation from elsewhere so old imports keep working.
Status: LEGACY / SHIM
Important notes for non-coders: Prefer importing the real module at package root or the named implementation file.
"""

from ..router_losses import router_regularizer, symmetric_kl

__all__ = ["router_regularizer", "symmetric_kl"]

