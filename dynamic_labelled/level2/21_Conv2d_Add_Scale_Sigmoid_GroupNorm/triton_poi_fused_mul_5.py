# From: 21_Conv2d_Add_Scale_Sigmoid_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mul_5poi_fused_mul_5(input_ptr0, input_ptr1, output_ptr0, kernel_size, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    linear_index = block_indices
    group_index = (block_indices // kernel_size) % 16
    input_data0 = tl.load(input_ptr0 + (linear_index), valid_mask, eviction_policy='evict_last')
    input_data1 = tl.load(input_ptr1 + (group_index), valid_mask, eviction_policy='evict_last')
    result_data = input_data0 * input_data1
    tl.store(output_ptr0 + (linear_index), result_data, valid_mask)