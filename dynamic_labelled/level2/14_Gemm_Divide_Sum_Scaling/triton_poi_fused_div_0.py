# From: 14_Gemm_Divide_Sum_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_0(in_ptr0, out_ptr0, num_elements, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    element_index = block_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = element_index < num_elements
    block_index = element_index // 20
    original_index = element_index
    input_values = tl.load(in_ptr0 + (block_index), valid_mask, eviction_policy='evict_last')
    scale_factor = 1.5
    scaled_values = input_values * scale_factor
    final_scale = 0.5
    result_values = scaled_values * final_scale
    tl.store(out_ptr0 + (original_index), result_values, valid_mask)