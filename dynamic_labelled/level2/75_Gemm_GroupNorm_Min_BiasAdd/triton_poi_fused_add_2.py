# From: 75_Gemm_GroupNorm_Min_BiasAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_2poi_fused_add_2(in_ptr0, in_ptr1, out_ptr0, kernel_size, num_elements, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = block_indices < num_elements
    kernel_index = block_indices % kernel_size
    batch_index = block_indices // kernel_size
    linear_index = block_indices
    input_data0 = tl.load(in_ptr0 + (kernel_index), valid_mask, eviction_policy='evict_last')
    input_data1 = tl.load(in_ptr1 + (batch_index), valid_mask, eviction_policy='evict_last')
    result_data = input_data0 + input_data1
    tl.store(out_ptr0 + (linear_index), result_data, valid_mask)