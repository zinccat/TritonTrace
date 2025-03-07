# From: 75_Gemm_GroupNorm_Min_BiasAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_2(in_ptr0, in_ptr1, out_ptr0, kernel_size, num_elements, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = block_indices < num_elements
    index_mod_kernel = block_indices % kernel_size
    index_div_kernel = block_indices // kernel_size
    linear_index = block_indices

    data_from_in_ptr0 = tl.load(in_ptr0 + (index_mod_kernel), valid_mask, eviction_policy='evict_last')
    data_from_in_ptr1 = tl.load(in_ptr1 + (index_div_kernel), valid_mask, eviction_policy='evict_last')
    result = data_from_in_ptr0 + data_from_in_ptr1

    tl.store(out_ptr0 + (linear_index), result, valid_mask)