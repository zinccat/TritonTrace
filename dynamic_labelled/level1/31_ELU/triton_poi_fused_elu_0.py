# From: 31_ELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_elu_0(in_ptr0, out_ptr0, num_elements, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    element_index = block_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = element_index < num_elements
    indices = element_index
    input_values = tl.load(in_ptr0 + (indices), valid_mask)
    zero_threshold = 0.0
    is_positive = input_values > zero_threshold
    one_multiplier = 1.0
    positive_values = input_values * one_multiplier
    expm1_values = tl.extra.cuda.libdevice.expm1(positive_values)
    scaled_expm1 = expm1_values * one_multiplier
    elu_result = tl.where(is_positive, positive_values, scaled_expm1)
    tl.store(out_ptr0 + (indices), elu_result, valid_mask)