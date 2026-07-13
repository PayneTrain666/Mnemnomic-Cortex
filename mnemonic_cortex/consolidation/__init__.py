"""
Plain-language summary
----------------------
What this file is for: Compatibility import shim for older paths under mnemonic_cortex/consolidation/.
How it fits in the system: Re-exports the real implementation from elsewhere so old imports keep working.
Status: LEGACY / SHIM
Important notes for non-coders: Prefer importing the real module at package root or the named implementation file.
"""

from .consolidated_memory_store import ConsolidatedMemoryCfg, ConsolidatedMemoryStore
from .consolidated_param_store import ConsolidatedParamStore, UnifiedParamCfg
from .consolidation_broker import BrokerCfg, ConsolidationBrokerV2
from .multi_cps import MultiCPSManager

