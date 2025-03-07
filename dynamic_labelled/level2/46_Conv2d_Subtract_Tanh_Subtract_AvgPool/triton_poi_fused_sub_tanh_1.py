# From: 46_Conv2d_Subtract_Tanh_Subtract_AvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_sub_tanh_1(in_ptr0, out_ptr0, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    input_indices = indices
    input_values = tl.load(in_ptr0 + (input_indices), mask)
    subtract_value = 0.5
    subtracted_values = input_values - subtract_value
    tanh_values = tl.extra.cuda.libdevice.tanh(subtracted_values)
    subtract_tanh_value = 0.2
    final_values = tanh_values - subtract_tanh_value
    tl.store(out_ptr0 + (input_indices), final_values, mask)