# From: 90_cumprod

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def _triton_helper_fn_mul0(input_value, multiplier):
    product = input_value * multiplier
    return product