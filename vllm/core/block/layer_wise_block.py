from typing import Dict, List, Optional, Set
from vllm.core.block.naive_block import NaiveBlockAllocator, NaiveBlock
from vllm.core.block.interfaces import Block, BlockAllocator
from vllm.utils import Device

class LayerWiseBlockAllocator(NaiveBlockAllocator):
    """支持层级分配的Block分配器"""

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_gpu_blocks: int,
    ):
        super().__init__(
            create_block=LayerWiseBlock,
            num_blocks=num_blocks,
            block_size=block_size,
        )
        
        self.num_layers = num_layers
        self.num_gpu_blocks = num_gpu_blocks
        
        # 每层的blocks跟踪
        self.layer_blocks: Dict[int, Set[int]] = {
            i: set() for i in range(num_layers)
        }
        
        # 每层的GPU blocks跟踪  
        self.layer_gpu_blocks: Dict[int, Set[int]] = {
            i: set() for i in range(num_layers)
        }

    def allocate_layer_block(
        self,
        layer_id: int,
        prev_block: Optional[Block],
        token_ids: List[int],
        device: Device
    ) -> Block:
        """为指定层分配block"""
        block_id = self._allocate_block_id()
        
        block = LayerWiseBlock(
            prev_block=prev_block,
            token_ids=token_ids,
            block_size=self._block_size,
            allocator=self,
            block_id=block_id,
            layer_id=layer_id,
            device=device
        )
        
        # 记录分配
        self.layer_blocks[layer_id].add(block_id)
        if device == Device.GPU:
            self.layer_gpu_blocks[layer_id].add(block_id)
            
        return block

    def free_layer_block(self, block: LayerWiseBlock) -> None:
        """释放指定层的block"""
        layer_id = block.layer_id
        block_id = block.block_id
        
        if block_id is None:
            return
            
        # 清理记录
        self.layer_blocks[layer_id].remove(block_id)
        if block.device == Device.GPU:
            self.layer_gpu_blocks[layer_id].remove(block_id)
            
        super().free(block)

    def get_layer_gpu_blocks(self, layer_id: int) -> Set[int]:
        """获取指定层的GPU blocks"""
        return self.layer_gpu_blocks[layer_id]

    def get_layer_free_gpu_blocks(self, layer_id: int) -> int:
        """获取指定层的可用GPU blocks数量"""
        return self.num_gpu_blocks - len(self.layer_gpu_blocks[layer_id])


class LayerWiseBlock(NaiveBlock):
    """支持层级管理的Block实现"""
    
    def __init__(self,
                 prev_block: Optional[Block],
                 token_ids: List[int],
                 block_size: int,
                 allocator: BlockAllocator,
                 block_id: Optional[int] = None,
                 layer_id: int = 0,
                 device: Device = Device.GPU):
        super().__init__(
            prev_block=prev_block,
            token_ids=token_ids,
            block_size=block_size,
            allocator=allocator,
            block_id=block_id
        )
        self.layer_id = layer_id
        self.device = device
        self._computed = False
        self._last_accessed = 0.0

    @property
    def computed(self) -> bool:
        return self._computed

    @computed.setter 
    def computed(self, value: bool) -> None:
        self._computed = value

    @property
    def last_accessed(self) -> float:
        return self._last_accessed
    
    @last_accessed.setter
    def last_accessed(self, value: float) -> None:
        self._last_accessed = value