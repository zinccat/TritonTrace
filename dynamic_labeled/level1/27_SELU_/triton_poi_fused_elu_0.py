# From: 27_SELU_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_elu_0(in_ptr0, out_ptr0, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    base_indices = indices
    input_values = tl.load(in_ptr0 + (base_indices), mask)
    zero_threshold = 0.0
    is_positive = input_values > zero_threshold
    alpha = 1.0507009873554805
    scaled_positive = input_values * alpha
    one = 1.0
    scaled_input = input_values * one
    expm1_result = tl.extra.cuda.libdevice.expm1(scaled_input)
    beta = 1.7580993408473766
    scaled_expm1 = expm1_result * beta
    elu_result = tl.where(is_positive, scaled_positive, scaled_expm1)
    tl.store(out_ptr0 + (base_indices), elu_result, mask)