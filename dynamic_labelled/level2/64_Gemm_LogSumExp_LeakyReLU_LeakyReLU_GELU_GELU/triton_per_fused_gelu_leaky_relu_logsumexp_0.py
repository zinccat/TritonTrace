# From: 64_Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_gelu_leaky_relu_logsumexp_0(
    in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel
):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 512

    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    r_index = tl.arange(0, RBLOCK)[:]
    
    input_value = tl.load(in_ptr0 + (r_index + 512 * x_index), None)
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
    leaky_relu_threshold = 0.0
    leaky_relu_condition = log_sum_exp_stable > leaky_relu_threshold
    leaky_relu_slope = 0.01
    leaky_relu_output = tl.where(leaky_relu_condition, log_sum_exp_stable, log_sum_exp_stable * leaky_relu_slope)
    gelu_threshold = 0.0
    gelu_condition = leaky_relu_output > gelu_threshold
    gelu_slope = 0.01
    gelu_scaled_output = tl.where(gelu_condition, leaky_relu_output, leaky_relu_output * gelu_slope)
    gelu_half = 0.5
    gelu_sqrt_half = 0.7071067811865476
    erf_input = gelu_scaled_output * gelu_sqrt_half
    erf_output = tl.extra.cuda.libdevice.erf(erf_input)
    erf_one = 1.0
    erf_result = erf_output + erf_one
    gelu_result = gelu_half * erf_result * gelu_half
    erf_input_gelu = gelu_scaled_output * gelu_sqrt_half
    erf_output_gelu = tl.extra.cuda.libdevice.erf(erf_input_gelu)
    erf_result_gelu = erf_output_gelu + erf_one
    final_gelu_output = gelu_result * erf_result_gelu

    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x_index), log_sum_exp_stable, None)
    tl.store(out_ptr1 + (x_index), final_gelu_output, None)