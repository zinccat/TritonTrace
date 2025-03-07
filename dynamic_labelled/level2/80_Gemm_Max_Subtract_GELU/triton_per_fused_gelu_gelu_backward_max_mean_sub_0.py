# From: 80_Gemm_Max_Subtract_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_gelu_gelu_backward_max_mean_sub_0(in_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 1024
    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[:]
    tl.full([RBLOCK], True, tl.int1)
    r1 = r_index
    x0 = x_index
    input_tensor = tl.load(in_ptr0 + (r1 + 1024 * x0), None)
    broadcast_input = tl.broadcast_to(input_tensor, [RBLOCK])
    max_value = triton_helpers.promote_to_tensor(triton_helpers.max2(broadcast_input, 0))
    r_index_broadcast = tl.broadcast_to(r_index, broadcast_input.shape)
    max_value_val, max_value_idx = triton_helpers.max_with_index(broadcast_input, r_index_broadcast, 0)
    max_value_index = triton_helpers.promote_to_tensor(max_value_idx)
    divisor = 1.0
    normalized_max = max_value / divisor
    subtracted_max = max_value - normalized_max
    half = 0.5
    scaled_subtracted_max = subtracted_max * half
    sqrt_half = 0.7071067811865476
    erf_input = subtracted_max * sqrt_half
    erf_result = tl.extra.cuda.libdevice.erf(erf_input)
    erf_adjusted = erf_result + divisor
    gelu_result = scaled_subtracted_max * erf_adjusted
    erf_scaled = erf_adjusted * half
    squared_subtracted_max = subtracted_max * subtracted_max
    neg_half = -0.5
    exp_input = squared_subtracted_max * neg_half
    exp_result = tl.math.exp(exp_input)
    sqrt_two_pi = 0.3989422804014327
    exp_scaled = exp_result * sqrt_two_pi
    exp_adjusted = subtracted_max * exp_scaled
    final_result = erf_scaled + exp_adjusted
    tl.store(out_ptr2 + (x0), gelu_result, None)
    tl.store(out_ptr3 + (x0), final_result, None)
    tl.store(out_ptr1 + (x0), max_value_index, None)