"""CacheEngine class for managing the KV cache."""
from typing import List, Dict, Optional

import torch

from vllm.attention import get_attn_backend
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig, ParallelConfig,
                         LayerKVConfig)
from vllm.logger import init_logger
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType,
                        get_dtype_size, is_pin_memory_available)
from vllm.worker.cache_engine import CacheEngine

logger = init_logger(__name__)

class LayerWiseCacheEngine(CacheEngine):
    """Manages the KV cache by layer.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches for each layer. It also provides methods for performing KV cache
    operations, such as swapping and copying, on a per-layer basis.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
        layer_kv_config: LayerKVConfig,
        paramaters: int,
    ) -> None:
        assert cache_config.enable_layer_wise_cache
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

        self.layer_kv_config = layer_kv_config

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks , self.block_size, self.num_kv_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: torch.Tensor = torch.zeros(kv_cache_shape, dtype=self.dtype,
                                             pin_memory=pin_memory, device=device)
        return kv_cache

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
            self.attn_backend.swap_blocks(self.cpu_cache,
                                          self.gpu_cache, src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
            self.attn_backend.swap_blocks(self.gpu_cache,
                                          self.cpu_cache, src_to_dst)

    def copy(self, layer_idx: int, src_to_dsts: torch.Tensor) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache[layer_idx], src_to_dsts)

    def calculate_minimum_gpu_layers(self, seqlen: int,
                                     Tprefill: Optional[float] = None)-> int:
        """计算需要保留在GPU中的最小层数"""
        L = self.num_attention_layers
        dheads = self.head_size
        nheads = self.num_kv_heads
        fprecision = get_dtype_size(self.dtype)
        
        # 计算预填充时间（使用超线性关系）
        if not Tprefill:
            Tprefill = self._calculate_prefill_time(seqlen)
        
        # 二分查找最小的x值满足条件
        left, right = 0, L
        while left < right:
            x = (left + right) // 2
            Toffload = self._calculate_offload_time(seqlen, L, x, dheads, 
                                                  nheads, fprecision)
            if Toffload <= Tprefill:
                right = x
            else:
                left = x + 1
        return left

    def _calculate_prefill_time(self, seqlen: int) -> float:
        """计算预填充时间"""
        # 根据序列长度的超线性关系估算
        return (self.alpha * seqlen * \
            (2 * self.paramaters + 2 * seqlen * self.num_attention_layers)\
            / self.FLOP)

    def _calculate_offload_time(self, seqlen: int, L: int, x: int, 
                              dheads: int, nheads: int, fprecision: float) -> float:
        """计算卸载时间"""
        return (self.beta * seqlen * 2 * (L - x) * dheads * nheads * 
                fprecision / self.pcie_bandwidth)

    def update_gpu_blocks_status(self, t: int) -> None:
        """更新GPU块状态"""
        self.available_gpu_blocks = 100
        return
        released = self._estimate_released_blocks(t)
        allocated = self._estimate_allocated_blocks(t)
        self.available_gpu_blocks = (self.available_gpu_blocks + 
                                   released - allocated)
        
        # 检查是否需要卸载
        if self.available_gpu_blocks < self.cache_config.min_gpu_blocks:
            self._trigger_offload()

    def _estimate_released_blocks(self, t: int) -> int:
        """估算将要释放的块数"""
        # TODO: 实现基于多类预测模型的序列完成估算
        return sum(layer.get_completed_sequences_blocks() 
                  for layer in self.layer_states.values())

    def _estimate_allocated_blocks(self, t: int) -> int:
        """估算将要分配的块数"""
        # 保守估计：每个序列需要一个额外的KV块
        return sum(layer.get_running_sequences() 
                  for layer in self.layer_states.values())

    def _trigger_offload(self) -> None:
        """触发KV块卸载"""
        # 首先尝试卸载一半的层
        layers_to_offload = len(self.layer_states) // 2
        if not self._offload_layers(layers_to_offload):
            # 如果不够，执行完整卸载
            self._offload_layers(len(self.layer_states))

    def _offload_layers(self, num_layers: int) -> bool:
        """卸载指定数量的层到CPU"""
        # 优先卸载最近处理的请求
        recent_layers = sorted(self.layer_states.items(), 
                             key=lambda x: x[1].last_access_time,
                             reverse=True)[:num_layers]
        
        for layer_id, layer_state in recent_layers:
            # 异步卸载到CPU
            self._offload_layer_to_cpu(layer_id, layer_state)
        
        return True

    async def _offload_layer_to_cpu(self, layer_id: int, 
                                   layer_state: 'LayerState') -> None:
        """异步将层卸载到CPU"""
        blocks_to_offload = layer_state.get_blocks()
        for block in blocks_to_offload:
            await self.swap_out(block)
        self.available_gpu_blocks += len(blocks_to_offload)
        
    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = key_cache_block + value_cache_block
        # num_attention_layers = model_config.get_num_layers_by_block_type(
        #     parallel_config, LayerBlockType.attention)

        # key_cache_block = cache_config.block_size * num_heads * head_size
        # value_cache_block = key_cache_block
        # total = num_attention_layers * (key_cache_block + value_cache_block)
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = get_dtype_size(dtype)
        return dtype_size * total

class LayerState:
    def __init__(self):
        self.blocks = set()  # 当前层使用的块

    def add_block(self, block_id: int) -> None:
        self.blocks.add(block_id)

    def remove_block(self, block_id: int) -> None:
        self.blocks.remove(block_id)

    def get_blocks(self) -> set[int]:
        return self.blocks