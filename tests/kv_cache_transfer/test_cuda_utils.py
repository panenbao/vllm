from vllm.attention.backends.flash_attn import FlashAttentionBackend
from vllm.model_executor.cuda_utils import KVCacheTransferManager
import torch
from typing import List
import pytest


def _allocate_kv_cache(
        num_blocks: int,
        device: str,
        dtype: torch.dtype,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = FlashAttentionBackend.get_kv_cache_shape(
            num_blocks , 16, 32, 128)
        pin_memory = True if device == "cpu" else False
        kv_cache: torch.Tensor = torch.zeros(kv_cache_shape, dtype=dtype,
                                             pin_memory=pin_memory, device=device)
        return kv_cache

@pytest.mark.parametrize("dtype", [torch.half,
                                   torch.bfloat16,
                                   torch.float,
                                   torch.uint8])
def test_transfer_tensor(dtype):
    gpu_tensor = _allocate_kv_cache(
        num_blocks=100,
        device="cuda",
        dtype=dtype,
    )
    cpu_tensor = _allocate_kv_cache(
        num_blocks=100,
        device="cpu",
        dtype=dtype,
    )
    block_mapping = [[1,1],[1,2]]
    block_mapping = torch.tensor(block_mapping, device="cpu")
    transfer_manager = KVCacheTransferManager()
    transfer_manager.offload(gpu_tensor, cpu_tensor, block_mapping)
    transfer_manager.wait_transfer()
    

test_transfer_tensor(torch.float16)