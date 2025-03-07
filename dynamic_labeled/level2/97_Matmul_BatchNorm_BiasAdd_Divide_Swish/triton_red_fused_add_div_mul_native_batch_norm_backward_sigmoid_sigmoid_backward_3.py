# From: 97_Matmul_BatchNorm_BiasAdd_Divide_Swish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_div_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_3(
    input_grad_ptr, input_data_ptr, input_mean_ptr, input_var_ptr, input_scale_ptr,
    output_grad_ptr, output_mean_ptr, output_var_ptr, xnumel, rnumel, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):

    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex

    # Temporary storage for accumulated gradients
    accumulated_grad = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    accumulated_var_grad = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex

        # Load input gradients and data
        grad_input = tl.load(input_grad_ptr + (x0 + 512 * r1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        data_input = tl.load(input_data_ptr + (x0 + 512 * r1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        mean_input = tl.load(input_mean_ptr + (x0 + 512 * r1), rmask & xmask, eviction_policy='evict_first', other=0.0)

        # Sigmoid and its derivative
        sigmoid_output = tl.sigmoid(data_input)
        grad_sigmoid = grad_input * sigmoid_output
        grad_sigmoid_input = grad_input * data_input
        sigmoid_derivative = 1.0 - sigmoid_output
        grad_sigmoid_derivative = grad_sigmoid_input * sigmoid_derivative * sigmoid_output

        # Accumulate gradients
        grad_combined = grad_sigmoid + grad_sigmoid_derivative
        grad_broadcast = tl.broadcast_to(grad_combined, [XBLOCK, RBLOCK])
        accumulated_grad = tl.where(rmask & xmask, accumulated_grad + grad_broadcast, accumulated_grad)

        # Accumulate variance gradients
        var_grad = mean_input - tl.load(input_var_ptr + (x0), xmask, eviction_policy='evict_last')
        var_grad_broadcast = tl.broadcast_to(grad_combined * var_grad, [XBLOCK, RBLOCK])
        accumulated_var_grad = tl.where(rmask & xmask, accumulated_var_grad + var_grad_broadcast, accumulated_var_grad)

    # Sum over the second dimension
    output_grad_sum = tl.sum(accumulated_grad, 1)[:, None]
    output_var_grad_sum = tl.sum(accumulated_var_grad, 1)[:, None]

    # Store results
    tl.store(output_grad_ptr + (x0), output_grad_sum, xmask)
    tl.store(output_mean_ptr + (x0), output_grad_sum, xmask)
    scale_input = tl.load(input_scale_ptr + (x0), xmask, eviction_policy='evict_last')
    output_var_scaled = output_var_grad_sum * scale_input
    tl.store(output_var_ptr + (x0), output_var_scaled, xmask)