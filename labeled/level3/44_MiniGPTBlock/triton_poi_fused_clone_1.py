# From: 44_MiniGPTBlock

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_1poi_fused_clone_1(input_ptr0, input_ptr1, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_index = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    dim1_index = block_index % 96
    dim2_index = (block_index // 96) % 512
    dim3_index = (block_index // 49152) % 8
    dim4_index = block_index // 393216
    linear_index = block_index
    
    input_offset0 = dim1_index + 96 * dim3_index + 2304 * dim2_index + 1179648 * dim4_index
    input_offset1 = dim1_index + 96 * dim3_index
    
    temp0 = tl.load(input_ptr0 + input_offset0, None)
    temp1 = tl.load(input_ptr1 + input_offset1, None, eviction_policy='evict_last')
    temp2 = temp0 + temp1
    
    tl.store(output_ptr0 + linear_index, temp2, None)