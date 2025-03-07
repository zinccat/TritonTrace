# From: 84_Gemm_BatchNorm_Scaling_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional__softmax_backward_data_mul_sum_0per_fused__native_batch_norm_legit_functional__softmax_backward_data_mul_sum_0(
    input_grad, input_data, input_mean, input_inv_var, input_gamma, input_beta, input_output, output_grad, output_sum, xnumel, rnumel):

    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 512

    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex

    grad_input = tl.load(input_grad + (r1 + 512 * x0), None)
    data_input = tl.load(input_data + (r1 + 512 * x0), None)
    mean_input = tl.load(input_mean + (r1 + 512 * x0), None)
    inv_var_input = tl.load(input_inv_var + (r1), None, eviction_policy='evict_last')
    gamma_input = tl.load(input_gamma + (r1), None, eviction_policy='evict_last')
    beta_input = tl.load(input_beta + (r1), None, eviction_policy='evict_last')
    output_input = tl.load(input_output + (r1), None, eviction_policy='evict_last')

    grad_scaled = grad_input * data_input
    grad_scaled_broadcast = tl.broadcast_to(grad_scaled, [RBLOCK])
    sum_grad_scaled = triton_helpers.promote_to_tensor(tl.sum(grad_scaled_broadcast, 0))
    neg_data_input = -data_input
    grad_input_adjusted = tl.extra.cuda.libdevice.fma(neg_data_input, sum_grad_scaled, grad_scaled)

    mean_adjusted = mean_input - inv_var_input
    mean_scaled = mean_adjusted * gamma_input
    mean_scaled_adjusted = mean_scaled * beta_input
    output_adjusted = mean_scaled_adjusted + output_input

    grad_output = grad_input_adjusted * output_adjusted
    grad_output_broadcast = tl.broadcast_to(grad_output, [RBLOCK])
    sum_grad_output = triton_helpers.promote_to_tensor(tl.sum(grad_output_broadcast, 0))

    tl.store(output_grad + (x0), sum_grad_scaled, None)
    tl.store(output_sum + (x0), sum_grad_output, None)