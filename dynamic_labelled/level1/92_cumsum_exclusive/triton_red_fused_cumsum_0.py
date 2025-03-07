# From: 92_cumsum_exclusive

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def _triton_helper_fn_add0(input_value, increment_value):
    result = input_value + increment_value
    return result