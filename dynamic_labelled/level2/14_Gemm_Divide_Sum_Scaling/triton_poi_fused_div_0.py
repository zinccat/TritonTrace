# From: 14_Gemm_Divide_Sum_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_0poi_fused_div_0(in_ptr0, out_ptr0, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    block_index = indices // 20
    element_index = indices
    input_data = tl.load(in_ptr0 + (block_index), mask, eviction_policy='evict_last')
    scale_factor = 1.5
    scaled_data = input_data * scale_factor
    final_scale = 0.5
    result_data = scaled_data * final_scale
    tl.store(out_ptr0 + (element_index), result_data, mask)