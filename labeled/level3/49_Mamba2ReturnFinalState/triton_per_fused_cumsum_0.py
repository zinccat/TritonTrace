# From: 49_Mamba2ReturnFinalState

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def _triton_helper_fn_add0(input_value1, input_value2):
    result_sum = input_value1 + input_value2
    return result_sum