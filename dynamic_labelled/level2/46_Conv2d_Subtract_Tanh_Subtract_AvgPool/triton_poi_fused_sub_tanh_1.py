# From: 46_Conv2d_Subtract_Tanh_Subtract_AvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_sub_tanh_1poi_fused_sub_tanh_1(in_ptr0, out_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    element_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = element_indices < num_elements
    indices = element_indices
    input_values = tl.load(in_ptr0 + (indices), valid_mask)
    subtract_value1 = 0.5
    subtracted_values = input_values - subtract_value1
    tanh_values = tl.extra.cuda.libdevice.tanh(subtracted_values)
    subtract_value2 = 0.2
    final_values = tanh_values - subtract_value2
    tl.store(out_ptr0 + (indices), final_values, valid_mask)