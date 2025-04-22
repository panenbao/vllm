import torch
from dataclasses import dataclass
from typing import Optional, Tuple


class KVCacheTransferManager:
    """管理 KV Cache 在 GPU 和 CPU 之间的异步传输"""

    def __init__(self) -> None:
        self.copy_stream = torch.cuda.Stream()
        self.copy_event = torch.cuda.Event()
        self.is_transferring: bool = False

    def copy(self, src: torch.Tensor, dst: torch.Tensor, block_mapping: torch.Tensor) -> None:
        """交换 GPU 和 CPU 之间的 KV cache"""
        if self.is_transferring:
            self.copy_stream.synchronize()
            self.is_transferring = False

        # 启动异步传输
        with torch.cuda.stream(self.copy_stream):
            self.is_transferring = True
            num_blocks = block_mapping.size()[0]
            for i in range(num_blocks):
                src_block_number = block_mapping[i][0]
                dst_block_number = block_mapping[i][1]
                src_block_k = src[0][src_block_number]
                dst_block_k = dst[0][dst_block_number]
                dst_block_k.copy_(src_block_k, non_blocking=True)
                src_block_v = src[1][src_block_number]
                dst_block_v = dst[1][dst_block_number]
                dst_block_v.copy_(src_block_v, non_blocking=True)
            self.copy_event.record(self.copy_stream)

    def transfer_state(self) -> bool:
        """非阻塞地检查传输状态"""
        if not self.is_transferring:
            return False
            
        if self.copy_event.query():
            self.is_transferring = False
            return True
        return False

    def wait_transfer(self):
        """阻塞等待传输完成"""
        if not self.is_transferring:
            return
        
        self.copy_stream.synchronize()
        self.is_transferring = False
        
