# From: 38_L1Norm_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_2(input_ptr0, input_ptr1, output_ptr0, kernel_size, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    index2 = block_indices
    index1 = block_indices // kernel_size
    data0 = tl.load(input_ptr0 + (index2), valid_mask, eviction_policy='evict_last')
    data1 = tl.load(input_ptr1 + (index1), valid_mask, eviction_policy='evict_last')
    result = data0 / data1
    tl.store(output_ptr0 + (index2), result, valid_mask)