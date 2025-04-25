import time
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple


class KVCacheTransferManager:
    """管理 KV Cache 在 GPU 和 CPU 之间的异步传输"""

    def __init__(self) -> None:
        self.streams: List[List[torch.cuda.Stream]]
        self.events: List[List[torch.cuda.Event]]

    def offload(self, gpu_cache: torch.Tensor, cpu_cache: torch.Tensor, block_mapping: torch.Tensor) -> None:
        """GPU -> CPU"""
        if block_mapping is None or block_mapping.size()[0] == 0:
            return
        num_blocks = block_mapping.size()[0]
        with open('/home/panenbao/vllm/logs/offload_len.log', 'a') as f:
            f.write(f'{num_blocks}\n')
        self.streams = [[torch.cuda.Stream() for _ in range(2)] for _ in range(num_blocks)]
        self.events = [[torch.cuda.Event() for _ in range(2)] for _ in range(num_blocks)]
        for i in range(num_blocks):
            gpu_block_number = block_mapping[i][0]
            cpu_block_number = block_mapping[i][1]

            gpu_block_k = gpu_cache[0][gpu_block_number]
            cpu_block_k = cpu_cache[0][cpu_block_number]
            gpu_block_v = gpu_cache[1][gpu_block_number]
            cpu_block_v = cpu_cache[1][cpu_block_number]

            with torch.cuda.stream(self.streams[i][0]):
                cpu_block_k.copy_(gpu_block_k, non_blocking=True)
                self.events[i][0].record(self.streams[i][0])
            
            with torch.cuda.stream(self.streams[i][1]):
                cpu_block_v.copy_(gpu_block_v, non_blocking=True)

    def prefetch(self, cpu_cache: torch.Tensor, gpu_cache: torch.Tensor, block_mapping: torch.Tensor,
                 events: List[List[torch.cuda.Event]]) -> None:
        """CPU -> GPU"""
        if block_mapping is None or block_mapping.size()[0] == 0:
            return

        event_len = len(events) if events else 0
        num_blocks = block_mapping.size()[0]
        min_wait_event = min(event_len, num_blocks)
        with open('/home/panenbao/vllm/logs/prefetch_len.log', 'a') as f:
            f.write(f'{num_blocks}\n')
        self.streams = [[torch.cuda.Stream() for _ in range(2)] for _ in range(num_blocks)]
        self.events = [[torch.cuda.Event() for _ in range(2)] for _ in range(num_blocks)]
        for i in range(num_blocks):
            gpu_block_number = block_mapping[i][0]
            cpu_block_number = block_mapping[i][1]

            gpu_block_k = gpu_cache[0][gpu_block_number]
            cpu_block_k = cpu_cache[0][cpu_block_number]
            gpu_block_v = gpu_cache[1][gpu_block_number]
            cpu_block_v = cpu_cache[1][cpu_block_number]

            with torch.cuda.stream(self.streams[i][0]):
                # if i < min_wait_event:
                #     events[i][0].wait(self.streams[i][0])
                gpu_block_k.copy_(cpu_block_k, non_blocking=True)
                self.events[i][0].record(self.streams[i][0])

            with torch.cuda.stream(self.streams[i][1]):
                # if i < min_wait_event:
                #     events[i][1].wait(self.streams[i][1])
                gpu_block_v.copy_(cpu_block_v, non_blocking=True)
                self.events[i][1].record(self.streams[i][1])
                
    def get_events(self) -> List[List[torch.cuda.Event]]:
        """获取事件列表"""
        return self.events