import time
import numpy as np
import torch
from typing import List, Tuple
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def measure_prefill_time(
    seq_lens: List[int],
    model: torch.nn.Module,
    device: torch.device
) -> List[float]:
    """测量不同序列长度的预填充时间
    
    Args:
        seq_lens: 要测试的序列长度列表
        model: 要测试的模型
        device: 运行设备
        
    Returns:
        每个序列长度对应的预填充时间列表
    """
    times = []
    for seq_len in seq_lens:
        # 生成随机输入
        inputs = torch.randint(0, 50000, (1, seq_len), device=device)
        
        # 预热
        for _ in range(3):
            with torch.no_grad():
                model(inputs)
        
        # 测量时间
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            model(inputs)
            
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        times.append(end - start)
        logger.info(f"Sequence length {seq_len}: {times[-1]:.4f}s")
        
    return times

def measure_offload_time(
    seq_lens: List[int],
    num_layers: int,
    head_dim: int,
    num_heads: int,
    dtype: torch.dtype
) -> List[float]:
    """测量不同序列长度的KV缓存卸载时间
    
    Args:
        seq_lens: 要测试的序列长度列表
        num_layers: 模型层数 
        head_dim: 注意力头维度
        num_heads: 注意力头数量
        dtype: 数据类型
        
    Returns:
        每个序列长度对应的卸载时间列表
    """
    times = []
    for seq_len in seq_lens:
        # 创建随机KV缓存
        kv_cache = torch.randn(
            num_layers,
            2,  # key and value
            seq_len, 
            num_heads,
            head_dim,
            dtype=dtype,
            device="cuda"
        )
        
        # 预热
        for _ in range(3):
            cpu_cache = kv_cache.cpu()
            del cpu_cache
            
        # 测量时间
        torch.cuda.synchronize() 
        start = time.perf_counter()
        
        cpu_cache = kv_cache.cpu()
        torch.cuda.synchronize()
        
        end = time.perf_counter()
        times.append(end - start)
        
        del cpu_cache
        logger.info(f"Sequence length {seq_len}: {times[-1]:.4f}s")
        
    return times

def calculate_beta(
    prefill_times: List[float],
    offload_times: List[float],
    seq_lens: List[int],
    num_layers: int
) -> float:
    """计算beta经验因子
    
    基于实际测量的时间和理论模型计算最优的beta值
    
    Args:
        prefill_times: 实测的预填充时间列表
        offload_times: 实测的卸载时间列表
        seq_lens: 对应的序列长度列表
        num_layers: 模型层数
        
    Returns:
        计算得到的beta值
    """
    betas = []
    for t_prefill, t_offload, seq_len in zip(prefill_times, offload_times, seq_lens):
        # 理论卸载时间公式中的beta项
        # t_offload = beta * seq_len * num_layers / pcie_bandwidth
        # 求解beta
        theoretical_time = t_prefill  # 我们希望卸载时间等于预填充时间
        actual_time = t_offload
        beta = theoretical_time / actual_time
        betas.append(beta)
        
    # 返回所有beta值的中位数作为最终结果
    beta = float(np.median(betas))
    logger.info(f"Calculated beta values: {betas}")
    logger.info(f"Final beta value: {beta:.4f}")
    return beta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Hugging Face model name")
    parser.add_argument("--seq-lens", type=int, nargs="+", 
                        default=[128, 256, 512, 1024],
                        help="Sequence lengths to test")
    args = parser.parse_args()
    
    # 加载模型
    # from transformers import AutoModelForCausalLM
    from vllm.model_executor.model_loader import get_model
    from vllm.config import VllmConfig, ModelConfig
    model = get_model(vllm_config=VllmConfig())
    device = next(model.parameters()).device
    
    config = model.config
    num_layers = config.num_hidden_layers
    head_dim = config.hidden_size // config.num_attention_heads
    num_heads = config.num_attention_heads
    dtype = next(model.parameters()).dtype
    
    # 测量时间
    prefill_times = measure_prefill_time(args.seq_lens, model, device)
    offload_times = measure_offload_time(
        args.seq_lens, num_layers, head_dim, num_heads, dtype)
        
    # 计算beta
    beta = calculate_beta(prefill_times, offload_times, args.seq_lens, num_layers)
    
    print(f"\nRecommended beta value: {beta:.4f}")

if __name__ == "__main__":
    main()