# From: 84_Gemm_BatchNorm_Scaling_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional__softmax_backward_data_mul_sum_0(
    input_grad_ptr, input_data_ptr, input_scale_ptr, input_mean_ptr, input_var_ptr, input_inv_std_ptr, input_output_ptr, 
    output_grad_ptr, output_sum_ptr, xnumel, rnumel):

    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 512

    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex

    grad_input = tl.load(input_grad_ptr + (r1 + 512 * x0), None)
    data_input = tl.load(input_data_ptr + (r1 + 512 * x0), None)
    scale_input = tl.load(input_scale_ptr + (r1 + 512 * x0), None)
    mean_input = tl.load(input_mean_ptr + (r1), None, eviction_policy='evict_last')
    var_input = tl.load(input_var_ptr + (r1), None, eviction_policy='evict_last')
    inv_std_input = tl.load(input_inv_std_ptr + (r1), None, eviction_policy='evict_last')
    output_input = tl.load(input_output_ptr + (r1), None, eviction_policy='evict_last')

    grad_scaled = grad_input * data_input
    grad_scaled_broadcast = tl.broadcast_to(grad_scaled, [RBLOCK])
    sum_grad_scaled = triton_helpers.promote_to_tensor(tl.sum(grad_scaled_broadcast, 0))
    neg_data_input = -data_input
    grad_input_adjusted = tl.extra.cuda.libdevice.fma(neg_data_input, sum_grad_scaled, grad_scaled)

    mean_diff = scale_input - mean_input
    var_scaled = mean_diff * var_input
    var_scaled_adjusted = var_scaled * inv_std_input
    output_adjusted = var_scaled_adjusted + output_input

    grad_output = grad_input_adjusted * output_adjusted
    grad_output_broadcast = tl.broadcast_to(grad_output, [RBLOCK])
    sum_grad_output = triton_helpers.promote_to_tensor(tl.sum(grad_output_broadcast, 0))

    tl.store(output_grad_ptr + (x0), sum_grad_scaled, None)
    tl.store(output_sum_ptr + (x0), sum_grad_output, None)