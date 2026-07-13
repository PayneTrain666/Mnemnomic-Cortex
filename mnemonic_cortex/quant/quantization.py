"""
Plain-language summary
----------------------
What this file is for: Compatibility import shim for older paths under mnemonic_cortex/quant/.
How it fits in the system: Re-exports the real implementation from elsewhere so old imports keep working.
Status: LEGACY / SHIM
Important notes for non-coders: Prefer importing the real module at package root or the named implementation file.
"""

from ..quantization import CPSQuantizer, QuantPolicy, dequant_int8_sym, quantize_int8_sym, wrap_angle

__all__ = ["CPSQuantizer", "QuantPolicy", "quantize_int8_sym", "dequant_int8_sym", "wrap_angle"]

