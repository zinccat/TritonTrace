# From: 64_Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_gelu_leaky_relu_logsumexp_0(in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    tl.full([RBLOCK], True, tl.int1)
    row_indices = rindex
    col_index = xindex
    input_value = tl.load(in_ptr0 + (row_indices + 512 * col_index), None)
    broadcast_input = tl.broadcast_to(input_value, [RBLOCK])
    max_value = triton_helpers.promote_to_tensor(triton_helpers.max2(broadcast_input, 0))
    abs_max_value = tl.math.abs(max_value)
    inf_value = float("inf")
    is_inf = abs_max_value == inf_value
    zero_value = 0.0
    stable_max = tl.where(is_inf, zero_value, max_value)
    shifted_input = input_value - stable_max
    exp_shifted = tl.math.exp(shifted_input)
    broadcast_exp = tl.broadcast_to(exp_shifted, [RBLOCK])
    sum_exp = triton_helpers.promote_to_tensor(tl.sum(broadcast_exp, 0))
    log_sum_exp = tl.math.log(sum_exp)
    log_sum_exp_stable = log_sum_exp + stable_max
    leaky_relu_threshold = 0.01
    leaky_relu = tl.where(log_sum_exp_stable > zero_value, log_sum_exp_stable, log_sum_exp_stable * leaky_relu_threshold)
    gelu_threshold = 0.01
    gelu = tl.where(leaky_relu > zero_value, leaky_relu, leaky_relu * gelu_threshold)
    gelu_coefficient = 0.5
    gelu_scaled = gelu * gelu_coefficient
    erf_coefficient = 0.7071067811865476
    erf_input = gelu * erf_coefficient
    erf_result = tl.extra.cuda.libdevice.erf(erf_input)
    erf_one = 1.0
    erf_sum = erf_result + erf_one
    gelu_erf = gelu_scaled * erf_sum
    gelu_erf_scaled = gelu_scaled * erf_coefficient
    erf_final = tl.extra.cuda.libdevice.erf(gelu_erf_scaled)
    erf_final_sum = erf_final + erf_one
    final_gelu = gelu_erf * erf_final_sum
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (col_index), log_sum_exp_stable, None)
    tl.store(out_ptr1 + (col_index), final_gelu, None)