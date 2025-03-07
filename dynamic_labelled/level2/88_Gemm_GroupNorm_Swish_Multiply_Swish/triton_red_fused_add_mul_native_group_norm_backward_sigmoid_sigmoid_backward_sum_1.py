# From: 88_Gemm_GroupNorm_Swish_Multiply_Swish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_mul_native_group_norm_backward_sigmoid_sigmoid_backward_sum_1red_fused_add_mul_native_group_norm_backward_sigmoid_sigmoid_backward_sum_1(
    input_grad_ptr, input_data_ptr, input_norm_ptr, input_scale_ptr, input_shift_ptr, input_shift_scale_ptr,
    output_grad_ptr0, output_grad_ptr1, output_grad_ptr2, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    input_norm = tl.load(input_norm_ptr + (x0), xmask, eviction_policy='evict_last')
    sum_grad_output = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    sum_grad_input = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    sum_grad_scale = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex // 64

    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        grad_output = tl.load(input_grad_ptr + (x0 + 1024 * r1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        input_data = tl.load(input_data_ptr + (x0 + 1024 * r1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        grad_scale = tl.load(input_scale_ptr + (x0 + 1024 * r1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        input_shift = tl.load(input_shift_ptr + (x3 + 16 * r1), rmask & xmask, eviction_policy='evict_last', other=0.0)
        input_shift_scale = tl.load(input_shift_scale_ptr + (x3 + 16 * r1), rmask & xmask, eviction_policy='evict_last', other=0.0)

        sigmoid_input_data = tl.sigmoid(input_data)
        grad_input_data = input_data * sigmoid_input_data
        grad_input_norm = grad_input_data * input_norm
        sigmoid_grad_input_norm = tl.sigmoid(grad_input_norm)
        grad_output_norm = grad_output * sigmoid_grad_input_norm
        grad_input_norm_squared = grad_output_norm * grad_input_norm
        one = 1.0
        one_minus_sigmoid = one - sigmoid_grad_input_norm
        grad_input_norm_complement = sigmoid_grad_input_norm * one_minus_sigmoid
        grad_input_norm_complement_squared = grad_input_norm_squared * grad_input_norm_complement
        grad_output_combined = grad_output_norm + grad_input_norm_complement_squared
        grad_output_combined_scaled = grad_output_combined * grad_input_data
        broadcast_grad_output_combined_scaled = tl.broadcast_to(grad_output_combined_scaled, [XBLOCK, RBLOCK])
        sum_grad_output += broadcast_grad_output_combined_scaled
        sum_grad_output = tl.where(rmask & xmask, sum_grad_output, sum_grad_output)

        grad_input_combined = grad_output_combined * input_norm
        grad_input_combined_scaled = grad_input_combined * sigmoid_input_data
        grad_input_combined_scaled_input_data = grad_input_combined * input_data
        one_minus_sigmoid_input_data = one - sigmoid_input_data
        sigmoid_input_data_complement = sigmoid_input_data * one_minus_sigmoid_input_data
        grad_input_combined_scaled_input_data_complement = grad_input_combined_scaled_input_data * sigmoid_input_data_complement
        grad_input_combined_scaled_combined = grad_input_combined_scaled + grad_input_combined_scaled_input_data_complement
        grad_input_combined_scaled_grad_scale = grad_input_combined_scaled_combined * grad_scale
        grad_input_combined_scaled_input_shift = grad_input_combined_scaled_combined * input_shift
        grad_input_combined_scaled_diff = grad_input_combined_scaled_grad_scale - grad_input_combined_scaled_input_shift
        grad_input_combined_scaled_diff_scaled = grad_input_combined_scaled_diff * input_shift_scale
        broadcast_grad_input_combined_scaled_diff_scaled = tl.broadcast_to(grad_input_combined_scaled_diff_scaled, [XBLOCK, RBLOCK])
        sum_grad_input += broadcast_grad_input_combined_diff_scaled
        sum_grad_input = tl.where(rmask & xmask, sum_grad_input, sum_grad_input)

        broadcast_grad_input_combined_scaled_combined = tl.broadcast_to(grad_input_combined_scaled_combined, [XBLOCK, RBLOCK])
        sum_grad_scale += broadcast_grad_input_combined_scaled_combined
        sum_grad_scale = tl.where(rmask & xmask, sum_grad_scale, sum_grad_scale)

    sum_grad_output_final = tl.sum(sum_grad_output, 1)[:, None]
    sum_grad_input_final = tl.sum(sum_grad_input, 1)[:, None]
    sum_grad_scale_final = tl.sum(sum_grad_scale, 1)[:, None]

    tl.store(output_grad_ptr0 + (x0), sum_grad_output_final, xmask)
    tl.store(output_grad_ptr1 + (x0), sum_grad_input_final, xmask)
    tl.store(output_grad_ptr2 + (x0), sum_grad_scale_final, xmask)