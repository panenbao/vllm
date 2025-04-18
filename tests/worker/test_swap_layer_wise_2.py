from vllm.attention.backends.flash_attn import FlashAttentionBackend
from vllm.utils import is_pin_memory_available
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig, ParallelConfig,
                         LayerKVConfig)
from vllm.engine.arg_utils import EngineArgs
from vllm.worker.layer_wise_cache_engine import LayerWiseCacheEngine
import torch

def test_swap():
    # Create the cache engine.
    engine_args = EngineArgs(model="/mnt/HDD0/panenbao/models/Llama-2-7b-hf",
                             dtype="half",)
    engine_config = engine_args.create_engine_config()
    engine_config.cache_config.num_gpu_blocks = 2200
    engine_config.cache_config.num_cpu_blocks = 1000
    engine = LayerWiseCacheEngine(cache_config=engine_config.cache_config,
                                  model_config=engine_config.model_config,
                                  parallel_config=engine_config.parallel_config,
                                  device_config=engine_config.device_config,
                                  layer_kv_config=engine_config.layer_kv_config,
                                  paramaters=0)
    gpu_cache = engine.gpu_cache
    cpu_cache = engine.cpu_cache
    print(gpu_cache.shape)
    print(cpu_cache.shape)
    gpu_cache[0][19] = torch.ones(16,32,128, dtype=torch.float16, device='cuda')
    cpu_cache[0][23] = torch.ones(16,32,128, dtype=torch.float16, device='cpu')
    blocks_to_swap_in = [
        (19, 45),
        (67, 23),
        (12, 78),
        (40, 99),
        (1, 71),
    ]
    blocks_to_swap_in = torch.tensor(blocks_to_swap_in, dtype=torch.float16)
    FlashAttentionBackend.swap_blocks(gpu_cache, cpu_cache, blocks_to_swap_in)
    # print(gpu_cache)
    # print(cpu_cache)
    
if __name__ == "__main__":
    test_swap()