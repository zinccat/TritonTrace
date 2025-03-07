# From: 22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_clamp_logsumexp_mish_mul_0per_fused_add_clamp_logsumexp_mish_mul_0(in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    tl.full([RBLOCK], True, tl.int1)
    row_indices = rindex
    col_indices = xindex
    input_value = tl.load(in_ptr0 + (row_indices + 1024 * col_indices), None)
    scale_factor = 2.0
    scaled_value = input_value * scale_factor
    doubled_value = scaled_value + scaled_value
    clamp_min = -10.0
    clamped_value = triton_helpers.maximum(doubled_value, clamp_min)
    clamp_max = 10.0
    clamped_and_clipped_value = triton_helpers.minimum(clamped_value, clamp_max)
    broadcast_clamped_value = tl.broadcast_to(clamped_and_clipped_value, [RBLOCK])
    max_value = triton_helpers.promote_to_tensor(triton_helpers.max2(broadcast_clamped_value, 0))
    abs_max_value = tl.math.abs(max_value)
    infinity = float("inf")
    is_infinity = abs_max_value == infinity
    zero_value = 0.0
    safe_max_value = tl.where(is_infinity, zero_value, max_value)
    adjusted_value = clamped_and_clipped_value - safe_max_value
    exp_adjusted_value = tl.math.exp(adjusted_value)
    broadcast_exp_value = tl.broadcast_to(exp_adjusted_value, [RBLOCK])
    sum_exp_values = triton_helpers.promote_to_tensor(tl.sum(broadcast_exp_value, 0))
    log_sum_exp = tl.math.log(sum_exp_values)
    final_value = log_sum_exp + safe_max_value
    log_threshold = 20.0
    is_large_value = final_value > log_threshold
    exp_final_value = tl.math.exp(final_value)
    log1p_exp_final_value = tl.extra.cuda.libdevice.log1p(exp_final_value)
    log_or_log1p_value = tl.where(is_large_value, final_value, log1p_exp_final_value)
    tanh_value = tl.extra.cuda.libdevice.tanh(log_or_log1p_value)
    product_value = final_value * tanh_value
    mish_value = final_value * product_value
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (col_indices), final_value, None)
    tl.store(out_ptr1 + (col_indices), mish_value, None)