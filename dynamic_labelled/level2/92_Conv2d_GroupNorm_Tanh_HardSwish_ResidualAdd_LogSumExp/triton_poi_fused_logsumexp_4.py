# From: 92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_logsumexp_4poi_fused_logsumexp_4(in_out_ptr0, in_ptr0, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    base_indices = indices
    loaded_values = tl.load(in_out_ptr0 + (base_indices), mask)
    input_values = tl.load(in_ptr0 + (base_indices), mask)
    log_values = tl.math.log(loaded_values)
    abs_values = tl.math.abs(input_values)
    inf_value = float("inf")
    is_inf_mask = abs_values == inf_value
    zero_value = 0.0
    adjusted_values = tl.where(is_inf_mask, zero_value, input_values)
    result_values = log_values + adjusted_values
    tl.store(in_out_ptr0 + (base_indices), result_values, mask)