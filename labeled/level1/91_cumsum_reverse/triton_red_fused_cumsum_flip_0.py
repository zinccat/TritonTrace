# From: 91_cumsum_reverse

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def _triton_helper_fn_add0(input_value, increment_value):
    result = input_value + increment_value
    return result