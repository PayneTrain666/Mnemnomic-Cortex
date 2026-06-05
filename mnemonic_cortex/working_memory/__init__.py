"""Working memory module with context compression and episodic memory."""
import json
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any, Tuple


@dataclass
class ContextCompressionConfig:
    """Configuration for context compression."""
    dim: int = 16
    top_k_context_tokens: int = 3
    top_k_response_tokens: int = 2
    max_parameter_refs: int = 8


@dataclass
class SharedSlotStoreConfig:
    """Configuration for shared slot store."""
    namespace: str = "default"
    dim: int = 16
    max_slots: int = 1000


@dataclass
class QuantumHolographicStorageConfig:
    """Configuration for quantum holographic storage."""
    dim: int = 16
    num_depths: int = 8
    max_records: int = 10000


class CompressionOutput:
    """Output from context compressor."""
    def __init__(self, compressed_context, compressed_response, episode_vector, context_top_indices, response_top_indices):
        self.compressed_context = compressed_context
        self.compressed_response = compressed_response
        self.episode_vector = episode_vector
        self.context_top_indices = context_top_indices
        self.response_top_indices = response_top_indices
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'compressed_context': self.compressed_context.cpu().detach().tolist() if torch.is_tensor(self.compressed_context) else self.compressed_context,
            'compressed_response': self.compressed_response.cpu().detach().tolist() if torch.is_tensor(self.compressed_response) else self.compressed_response,
            'episode_vector': self.episode_vector.cpu().detach().tolist() if torch.is_tensor(self.episode_vector) else self.episode_vector,
            'context_top_indices': self.context_top_indices.cpu().detach().tolist() if torch.is_tensor(self.context_top_indices) else self.context_top_indices,
            'response_top_indices': self.response_top_indices.cpu().detach().tolist() if torch.is_tensor(self.response_top_indices) else self.response_top_indices,
        }


class ParameterReference:
    """Reference to a model parameter."""
    def __init__(self, parameter_name: str, reference_kind: str, weight_magnitude: float):
        self.parameter_name = parameter_name
        self.reference_kind = reference_kind
        self.weight_magnitude = weight_magnitude
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'parameter_name': self.parameter_name,
            'reference_kind': self.reference_kind,
            'weight_magnitude': self.weight_magnitude,
            'paamax_metadata': {
                'full_weight_tensor_stored': False,
                'parameter_mutation': False,
            }
        }


class ContextCompressor(nn.Module):
    """Context compressor for reducing context and response to compressed representations."""
    
    def __init__(self, config: ContextCompressionConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.top_k_context = config.top_k_context_tokens
        self.top_k_response = config.top_k_response_tokens
        
        # Projection for importance scoring
        self.context_scorer = nn.Linear(config.dim, 1)
        self.response_scorer = nn.Linear(config.dim, 1)
        
        # Compression projection
        self.compress = nn.Linear(config.dim, config.dim)
    
    def forward(self, context: torch.Tensor, response: torch.Tensor) -> CompressionOutput:
        """Compress context and response.
        
        Args:
            context: (B, context_len, dim)
            response: (B, response_len, dim)
        
        Returns:
            CompressionOutput
        """
        B = context.size(0)
        
        # Score context tokens
        context_scores = self.context_scorer(context).squeeze(-1)  # (B, context_len)
        _, context_top_idx = torch.topk(context_scores, min(self.top_k_context, context.size(1)), dim=1)
        context_top_idx = context_top_idx.sort(dim=1)[0]  # (B, top_k)
        
        # Score response tokens
        response_scores = self.response_scorer(response).squeeze(-1)  # (B, response_len)
        _, response_top_idx = torch.topk(response_scores, min(self.top_k_response, response.size(1)), dim=1)
        response_top_idx = response_top_idx.sort(dim=1)[0]  # (B, top_k)
        
        # Extract top tokens
        batch_idx = torch.arange(B, device=context.device).unsqueeze(1)
        context_selected = context[batch_idx, context_top_idx]  # (B, top_k_context, dim)
        response_selected = response[batch_idx, response_top_idx]  # (B, top_k_response, dim)
        
        # Compress to single vectors
        compressed_context = context_selected.mean(dim=1)  # (B, dim)
        compressed_response = response_selected.mean(dim=1)  # (B, dim)
        
        # Create episode vector (combined representation)
        episode_vector = self.compress(compressed_context + compressed_response)  # (B, dim)
        
        return CompressionOutput(
            compressed_context=compressed_context,
            compressed_response=compressed_response,
            episode_vector=episode_vector,
            context_top_indices=context_top_idx,
            response_top_indices=response_top_idx,
        )


class ContextParameterReferenceExtractor(nn.Module):
    """Extract relevant parameter references from a model."""
    
    def __init__(self, config: ContextCompressionConfig):
        super().__init__()
        self.config = config
        self.max_refs = config.max_parameter_refs
    
    def extract(self, model: nn.Module, parameter_hints: Optional[List[str]] = None) -> List[ParameterReference]:
        """Extract top parameter references from model.
        
        Args:
            model: PyTorch model
            parameter_hints: List of keywords to filter parameters
        
        Returns:
            List of ParameterReference objects
        """
        refs = []
        parameter_hints = parameter_hints or []
        
        for name, param in model.named_parameters():
            # Filter by hints if provided
            if parameter_hints and not any(hint in name.lower() for hint in parameter_hints):
                continue
            
            magnitude = param.data.abs().mean().item()
            
            # Determine reference kind
            if 'weight' in name:
                kind = 'weight'
            elif 'bias' in name:
                kind = 'bias'
            else:
                kind = 'parameter'
            
            refs.append(ParameterReference(name, kind, magnitude))
        
        # Sort by magnitude and take top-k
        refs.sort(key=lambda r: r.weight_magnitude, reverse=True)
        refs = refs[:self.max_refs]
        
        return refs


@dataclass
class MemoryCandidate:
    """Candidate for storage in episodic memory."""
    context_map: str
    response_fingerprint: str
    weight_refs: List[ParameterReference] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    paamax_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.metadata:
            self.metadata = {'response_linked': True}
        if not self.paamax_metadata:
            self.paamax_metadata = {'memory_store_mutation_by_default': False}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'context_map': self.context_map,
            'response_fingerprint': self.response_fingerprint,
            'parameter_refs': [ref.to_dict() for ref in self.weight_refs],
            'metadata': self.metadata,
            'paamax_metadata': self.paamax_metadata,
        }


@dataclass
class MemoryBuildResult:
    """Result from memory building."""
    candidate: MemoryCandidate
    stored: bool = False
    shared_slot_result: Dict[str, Any] = field(default_factory=dict)
    qh_record: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'candidate': self.candidate.to_dict(),
            'stored': self.stored,
            'shared_slot_result': self.shared_slot_result,
            'qh_record': self.qh_record,
        }


class ContextEpisodicMemoryBuilder(nn.Module):
    """Build episodic memory candidates from context and response."""
    
    def __init__(self, config: ContextCompressionConfig):
        super().__init__()
        self.config = config
        self.compressor = ContextCompressor(config)
        self.extractor = ContextParameterReferenceExtractor(config)
    
    def build_and_stage(
        self,
        context: torch.Tensor,
        response: torch.Tensor,
        model: nn.Module,
        project_id: str,
        chat_id: str,
        episode_id: str,
        context_map: str = "episodic",
        task_hints: Optional[List[str]] = None,
        task_mode: Optional[str] = None,
        allow_store: bool = False,
        write_permission: bool = False,
        shared_slot_store: Optional['SharedSlotStore'] = None,
        qh_storage: Optional['QuantumHolographicStorage'] = None,
    ) -> MemoryBuildResult:
        """Build and stage episodic memory.
        
        Args:
            context: Context tensor
            response: Response tensor
            model: Model to extract parameters from
            project_id: Project identifier
            chat_id: Chat identifier
            episode_id: Episode identifier
            context_map: Type of context map
            task_hints: Task hints for parameter extraction
            task_mode: Task mode
            allow_store: Whether storing is allowed
            write_permission: Whether write permission is granted
            shared_slot_store: Shared slot store instance
            qh_storage: Quantum holographic storage instance
        
        Returns:
            MemoryBuildResult
        """
        # Compress context and response
        compression = self.compressor(context, response)
        
        # Extract parameter references
        param_refs = self.extractor.extract(model, task_hints)
        
        # Create fingerprint
        response_fingerprint = f"fp-{hash(response.data_ptr()) % 10000}"
        
        # Create memory candidate
        candidate = MemoryCandidate(
            context_map=context_map,
            response_fingerprint=response_fingerprint,
            weight_refs=param_refs,
            metadata={'response_linked': True},
            paamax_metadata={'memory_store_mutation_by_default': False}
        )
        
        result = MemoryBuildResult(candidate=candidate, stored=False)
        
        # Store if permission granted
        if allow_store and write_permission and shared_slot_store is not None:
            # Store to shared slot store
            slot_id = f"css-{project_id}-{episode_id}"
            shared_slot_store.store(slot_id, compression.episode_vector)
            result.stored = True
            result.shared_slot_result = {'canonical_id': slot_id}
            
            # Store to quantum holographic storage if available
            if qh_storage is not None:
                qh_id = f"qhrec-{project_id}-{episode_id}"
                qh_storage.store(qh_id, compression.episode_vector)
                result.qh_record = {
                    'record_id': qh_id,
                    'code_schema': {
                        'notice': 'quantum_holographic_compatible_metadata_not_quantum_hardware_claim'
                    }
                }
        
        return result


class SharedSlotStore(nn.Module):
    """Shared slot store for storing compressed memories."""
    
    def __init__(self, config: SharedSlotStoreConfig):
        super().__init__()
        self.config = config
        self.storage = {}
    
    def store(self, slot_id: str, vector: torch.Tensor) -> None:
        """Store a vector in a slot."""
        self.storage[slot_id] = vector.clone().detach()
    
    def retrieve(self, slot_id: str) -> Optional[torch.Tensor]:
        """Retrieve a vector from a slot."""
        return self.storage.get(slot_id)


class QuantumHolographicStorage(nn.Module):
    """Quantum holographic storage for episodic memories."""
    
    def __init__(self, config: QuantumHolographicStorageConfig, shared_slot_store: Optional[SharedSlotStore] = None):
        super().__init__()
        self.config = config
        self.storage = {}
        self.shared_slot_store = shared_slot_store
    
    def store(self, record_id: str, vector: torch.Tensor) -> None:
        """Store a vector in quantum holographic format."""
        self.storage[record_id] = vector.clone().detach()
    
    def retrieve(self, record_id: str) -> Optional[torch.Tensor]:
        """Retrieve a vector from quantum holographic storage."""
        return self.storage.get(record_id)


class GeometryMountedContextBuffer(nn.Module):
    """Buffer for context memory with geometry mounting."""
    
    def __init__(self, dim: int = 16, num_depths: int = 8):
        super().__init__()
        self.config = ContextCompressionConfig(dim=dim)
        self.builder = ContextEpisodicMemoryBuilder(self.config)
        self.compressor = ContextCompressor(self.config)
    
    def build_context_memory_candidate(
        self,
        context: torch.Tensor,
        response: torch.Tensor,
        model: nn.Module,
        project_id: str,
        chat_id: str,
        episode_id: str,
        context_map: str = "episodic",
        task_hints: Optional[List[str]] = None,
    ) -> Tuple[MemoryCandidate, torch.Tensor, CompressionOutput]:
        """Build context memory candidate with compression.
        
        Returns:
            Tuple of (candidate, episode_vector, compression_output)
        """
        compression = self.compressor(context, response)
        
        param_refs = self.builder.extractor.extract(model, task_hints)
        
        response_fingerprint = f"fp-{hash(response.data_ptr()) % 10000}"
        
        candidate = MemoryCandidate(
            context_map=context_map,
            response_fingerprint=response_fingerprint,
            weight_refs=param_refs,
        )
        
        return candidate, compression.episode_vector, compression


def wm_context_compression_contract() -> Dict[str, Any]:
    """Return the working memory context compression contract."""
    return {
        'payload': {
            'context_compression': True,
        },
        'paamax_metadata': {
            'write_permission_required': True,
        }
    }


__all__ = [
    'ContextCompressionConfig',
    'SharedSlotStoreConfig',
    'QuantumHolographicStorageConfig',
    'ContextCompressor',
    'ContextParameterReferenceExtractor',
    'ContextEpisodicMemoryBuilder',
    'SharedSlotStore',
    'QuantumHolographicStorage',
    'GeometryMountedContextBuffer',
    'ParameterReference',
    'MemoryCandidate',
    'MemoryBuildResult',
    'CompressionOutput',
    'wm_context_compression_contract',
]
