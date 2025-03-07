# From: 22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_clamp_logsumexp_mish_mul_0(in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    tl.full([RBLOCK], True, tl.int1)
    row_indices = rindex
    col_indices = xindex
    input_values = tl.load(in_ptr0 + (row_indices + 1024 * col_indices), None)
    scale_factor = 2.0
    scaled_values = input_values * scale_factor
    doubled_values = scaled_values + scaled_values
    clamp_min = -10.0
    clamped_values = triton_helpers.maximum(doubled_values, clamp_min)
    clamp_max = 10.0
    clamped_and_clipped_values = triton_helpers.minimum(clamped_values, clamp_max)
    broadcast_clamped_values = tl.broadcast_to(clamped_and_clipped_values, [RBLOCK])
    positive_clamped_values = triton_helpers.promote_to_tensor(triton_helpers.max2(broadcast_clamped_values, 0))
    abs_values = tl.math.abs(positive_clamped_values)
    infinity = float("inf")
    is_infinity = abs_values == infinity
    zero_value = 0.0
    corrected_values = tl.where(is_infinity, zero_value, positive_clamped_values)
    adjusted_values = clamped_and_clipped_values - corrected_values
    exp_values = tl.math.exp(adjusted_values)
    broadcast_exp_values = tl.broadcast_to(exp_values, [RBLOCK])
    sum_exp_values = triton_helpers.promote_to_tensor(tl.sum(broadcast_exp_values, 0))
    log_sum_exp = tl.math.log(sum_exp_values)
    final_values = log_sum_exp + corrected_values
    log_threshold = 20.0
    is_large_value = final_values > log_threshold
    exp_final_values = tl.math.exp(final_values)
    log1p_exp_final_values = tl.extra.cuda.libdevice.log1p(exp_final_values)
    log_or_log1p_values = tl.where(is_large_value, final_values, log1p_exp_final_values)
    tanh_values = tl.extra.cuda.libdevice.tanh(log_or_log1p_values)
    intermediate_result = final_values * tanh_values
    final_result = final_values * intermediate_result
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (col_indices), final_values, None)
    tl.store(out_ptr1 + (col_indices), final_result, None)