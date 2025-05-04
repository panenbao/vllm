import torch
from dataclasses import dataclass
from typing import List, Optional


class KVCacheTransferManager:
    """管理 KV Cache 在 GPU 和 CPU 之间的异步传输（支持批处理）"""

    def __init__(self, batch_size: int = 32) -> None:
        self.batch_size = batch_size
        self.streams: List[torch.cuda.Stream] = []
        self.events: List[List[torch.cuda.Event]] = []

    def offload(self, gpu_cache: torch.Tensor, cpu_cache: torch.Tensor, block_mapping: torch.Tensor) -> None:
        """GPU -> CPU（按批次处理）"""
        if block_mapping is None or block_mapping.size(0) == 0:
            return

        num_blocks = block_mapping.size(0)
        num_batches = (num_blocks + self.batch_size - 1) // self.batch_size
        self.streams = [torch.cuda.Stream() for _ in range(2)]  # k和v各一个stream
        self.events = [[torch.cuda.Event() for _ in range(2)] for _ in range(num_blocks)]

        with open('/home/panenbao/vllm/logs/offload_len.log', 'a') as f:
            f.write(f'{num_blocks}\n')

        for batch_idx in range(num_batches):
            start = batch_idx * self.batch_size
            end = min((batch_idx + 1) * self.batch_size, num_blocks)

            with torch.cuda.stream(self.streams[0]):
                for i in range(start, end):
                    gpu_block_number = block_mapping[i][0]
                    cpu_block_number = block_mapping[i][1]
                    cpu_cache[0][cpu_block_number].copy_(gpu_cache[0][gpu_block_number], non_blocking=True)
                    self.events[i][0].record(self.streams[0])

            with torch.cuda.stream(self.streams[1]):
                for i in range(start, end):
                    gpu_block_number = block_mapping[i][0]
                    cpu_block_number = block_mapping[i][1]
                    cpu_cache[1][cpu_block_number].copy_(gpu_cache[1][gpu_block_number], non_blocking=True)
                    self.events[i][1].record(self.streams[1])

    def prefetch(self, cpu_cache: torch.Tensor, gpu_cache: torch.Tensor, block_mapping: torch.Tensor,
                 events: Optional[List[List[torch.cuda.Event]]] = None) -> None:
        """CPU -> GPU（按批次处理）"""
        if block_mapping is None or block_mapping.size(0) == 0:
            return

        num_blocks = block_mapping.size(0)
        num_batches = (num_blocks + self.batch_size - 1) // self.batch_size
        self.streams = [torch.cuda.Stream() for _ in range(2)]
        self.events = [[torch.cuda.Event() for _ in range(2)] for _ in range(num_blocks)]

        event_len = len(events) if events else 0
        min_wait_event = min(event_len, num_blocks)

        with open('/home/panenbao/vllm/logs/prefetch_len.log', 'a') as f:
            f.write(f'{num_blocks}\n')

        for batch_idx in range(num_batches):
            start = batch_idx * self.batch_size
            end = min((batch_idx + 1) * self.batch_size, num_blocks)

            with torch.cuda.stream(self.streams[0]):
                for i in range(start, end):
                    gpu_block_number = block_mapping[i][0]
                    cpu_block_number = block_mapping[i][1]
                    if i < min_wait_event and events:
                        events[i][0].wait(self.streams[0])
                    gpu_cache[0][gpu_block_number].copy_(cpu_cache[0][cpu_block_number], non_blocking=True)
                    self.events[i][0].record(self.streams[0])

            with torch.cuda.stream(self.streams[1]):
                for i in range(start, end):
                    gpu_block_number = block_mapping[i][0]
                    cpu_block_number = block_mapping[i][1]
                    if i < min_wait_event and events:
                        events[i][1].wait(self.streams[1])
                    gpu_cache[1][gpu_block_number].copy_(cpu_cache[1][cpu_block_number], non_blocking=True)
                    self.events[i][1].record(self.streams[1])

    def get_events(self) -> List[List[torch.cuda.Event]]:
        """获取事件列表"""
        return self.events
