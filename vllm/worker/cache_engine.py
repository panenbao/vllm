"""CacheEngine class for managing the KV cache."""
from typing import List, Dict, Optional

import torch

from vllm.attention import get_attn_backend
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig, ParallelConfig,
                         LayerKVConfig)
from vllm.logger import init_logger
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType,
                        get_dtype_size, is_pin_memory_available)

logger = init_logger(__name__)


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config

        self.head_size = model_config.get_head_size()
        # Models like Jamba, have mixed typed layers, E.g Mamba
        self.num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        if self.num_gpu_blocks:
            self.num_gpu_blocks //= parallel_config.pipeline_parallel_size
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        if self.num_cpu_blocks:
            self.num_cpu_blocks //= parallel_config.pipeline_parallel_size

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(self.head_size,
                                             model_config.dtype,
                                             cache_config.cache_dtype,
                                             self.block_size,
                                             model_config.is_attention_free)

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(
            self.num_gpu_blocks, self.device_config.device_type)
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        for _ in range(self.num_attention_layers):
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            kv_cache.append(
                torch.zeros(kv_cache_shape,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device))
        return kv_cache

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_attention_layers * (key_cache_block + value_cache_block)
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = get_dtype_size(dtype)
        return dtype_size * total


class LayerWiseCacheEngine:
    """Manages layer-wise KV cache in a single continuous tensor."""

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """Initialize the cache engine.
        
        Args:
            num_layers: Number of transformer layers
            num_blocks: Number of blocks in KV cache
            block_size: Size of each block
            num_kv_heads: Number of KV heads
            head_size: Size of each head
            dtype: Data type of KV cache
            device: Device to store KV cache
        """
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.dtype = dtype
        self.device = device

        # Create a single continuous KV cache tensor for all layers
        # Shape: [num_layers, 2, num_blocks, block_size, num_kv_heads, head_size]
        self.kv_cache = torch.empty(
            (num_layers, 2, num_blocks, block_size, num_kv_heads, head_size),
            dtype=dtype,
            device=device)

        # Block tables for each layer
        self.layer_block_tables = [{} for _ in range(num_layers)]
        # Cache slot mappings for each layer
        self.layer_slot_mappings = [[] for _ in range(num_layers)]
        
    def get_layer_kv_cache(self, layer_idx: int) -> torch.Tensor:
        """Get KV cache view for a specific layer.
        
        Args:
            layer_idx: Index of the layer
            
        Returns:
            The KV cache tensor slice for the specified layer
        """
        return self.kv_cache[layer_idx]
        
    def _swap_blocks(self, src_layer: int, dst_layer: int, src_blocks: List[int], 
                   dst_blocks: List[int]):
        """Swap blocks between two layers' KV caches."""
        for src_block, dst_block in zip(src_blocks, dst_blocks):
            # Swap key cache
            self.kv_cache[src_layer, 0, src_block].copy_(
                self.kv_cache[dst_layer, 0, dst_block])
            self.kv_cache[dst_layer, 0, dst_block].copy_(
                self.kv_cache[src_layer, 0, src_block])
            # Swap value cache
            self.kv_cache[src_layer, 1, src_block].copy_(
                self.kv_cache[dst_layer, 1, dst_block])
            self.kv_cache[dst_layer, 1, dst_block].copy_(
                self.kv_cache[src_layer, 1, src_block])

    def update_slot_mapping(self, layer_idx: int, slot_mapping: torch.Tensor):
        """Update slot mapping for a specific layer.
        
        Args:
            layer_idx: Index of the layer
            slot_mapping: New slot mapping tensor
        """
        self.layer_slot_mappings[layer_idx] = slot_mapping

    def update_block_tables(self, layer_idx: int, block_tables: Dict[int, List[int]]):
        """Update block tables for a specific layer.
        
        Args:
            layer_idx: Index of the layer
            block_tables: New block tables dictionary
        """
        self.layer_block_tables[layer_idx] = block_tables

    def copy_blocks(self, src_layer: int, dst_layer: int, 
                   src_to_dist_mapping: Dict[int, List[int]]):
        """Copy blocks between layers based on mapping.
        
        Args:
            src_layer: Source layer index
            dst_layer: Destination layer index
            src_to_dist_mapping: Mapping from source blocks to destination blocks
        """
        for src_block, dst_blocks in src_to_dist_mapping.items():
            for dst_block in dst_blocks:
                # Copy key cache
                self.kv_cache[dst_layer, 0, dst_block].copy_(
                    self.kv_cache[src_layer, 0, src_block])
                # Copy value cache
                self.kv_cache[dst_layer, 1, dst_block].copy_(
                    self.kv_cache[src_layer, 1, src_block])
                
    def get_slot_mapping(self, layer_idx: int) -> List[int]:
        """Get slot mapping for a specific layer."""
        return self.layer_slot_mappings[layer_idx]
        
    def get_block_tables(self, layer_idx: int) -> Dict[int, List[int]]:
        """Get block tables for a specific layer."""
        return self.layer_block_tables[layer_idx]


class LayerState:
    def __init__(self):
        self.blocks = set()  # 当前层使用的块
        self.last_access_time = 0
        self.running_sequences = 0
        self.completed_sequences = 0

    def add_block(self, block_id: int) -> None:
        self.blocks.add(block_id)
        self.last_access_time = time.time()

    def remove_block(self, block_id: int) -> None:
        self.blocks.remove(block_id)

    def get_blocks(self) -> set[int]:
        return self.blocks

    def get_running_sequences(self) -> int:
        return self.running_sequences

    def get_completed_sequences_blocks(self) -> int:
        return self.completed_sequences

    def mark_sequence_complete(self) -> None:
        self.completed_sequences += 1
        self.running_sequences -= 1

    def add_running_sequence(self) -> None:
        self.running_sequences += 1