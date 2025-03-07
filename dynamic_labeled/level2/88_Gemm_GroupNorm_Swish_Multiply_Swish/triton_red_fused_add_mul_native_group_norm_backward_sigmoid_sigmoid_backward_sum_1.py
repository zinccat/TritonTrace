# From: 88_Gemm_GroupNorm_Swish_Multiply_Swish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_mul_native_group_norm_backward_sigmoid_sigmoid_backward_sum_1(
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
    sum_grad_output0 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    sum_grad_output1 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    sum_grad_output2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex // 64

    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        input_grad = tl.load(input_grad_ptr + (x0 + 1024 * r1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        input_data = tl.load(input_data_ptr + (x0 + 1024 * r1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        input_scale = tl.load(input_scale_ptr + (x0 + 1024 * r1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        input_shift = tl.load(input_shift_ptr + (x3 + 16 * r1), rmask & xmask, eviction_policy='evict_last', other=0.0)
        input_shift_scale = tl.load(input_shift_scale_ptr + (x3 + 16 * r1), rmask & xmask, eviction_policy='evict_last', other=0.0)

        sigmoid_input_data = tl.sigmoid(input_data)
        grad_sigmoid_input_data = input_data * sigmoid_input_data
        grad_input_norm = grad_sigmoid_input_data * input_norm
        sigmoid_grad_input_norm = tl.sigmoid(grad_input_norm)
        grad_input_grad = input_grad * sigmoid_grad_input_norm
        grad_input_data = input_grad * grad_input_norm
        one = 1.0
        one_minus_sigmoid_grad_input_norm = one - sigmoid_grad_input_norm
        grad_sigmoid_grad_input_norm = grad_input_norm * one_minus_sigmoid_grad_input_norm
        grad_input_grad_combined = grad_input_grad + grad_input_data * grad_sigmoid_grad_input_norm
        grad_output0 = grad_input_grad_combined * grad_sigmoid_input_data
        broadcast_grad_output0 = tl.broadcast_to(grad_output0, [XBLOCK, RBLOCK])
        sum_grad_output0 += broadcast_grad_output0
        sum_grad_output0 = tl.where(rmask & xmask, sum_grad_output0, sum_grad_output0)

        grad_output1 = grad_input_grad_combined * input_norm
        grad_output1 *= sigmoid_input_data
        grad_output1 *= input_data
        one_minus_sigmoid_input_data = one - sigmoid_input_data
        grad_sigmoid_input_data_combined = sigmoid_input_data * one_minus_sigmoid_input_data
        grad_output1_combined = grad_output1 * grad_sigmoid_input_data_combined
        grad_output1_combined += grad_output1 * sigmoid_input_data
        grad_output1_combined *= input_scale
        grad_output1_combined -= grad_output1_combined * input_shift
        grad_output1_combined *= input_shift_scale
        broadcast_grad_output1 = tl.broadcast_to(grad_output1_combined, [XBLOCK, RBLOCK])
        sum_grad_output1 += broadcast_grad_output1
        sum_grad_output1 = tl.where(rmask & xmask, sum_grad_output1, sum_grad_output1)

        broadcast_grad_output2 = tl.broadcast_to(grad_input_grad_combined, [XBLOCK, RBLOCK])
        sum_grad_output2 += broadcast_grad_output2
        sum_grad_output2 = tl.where(rmask & xmask, sum_grad_output2, sum_grad_output2)

    sum_grad_output0 = tl.sum(sum_grad_output0, 1)[:, None]
    sum_grad_output1 = tl.sum(sum_grad_output1, 1)[:, None]
    sum_grad_output2 = tl.sum(sum_grad_output2, 1)[:, None]

    tl.store(output_grad_ptr0 + (x0), sum_grad_output0, xmask)
    tl.store(output_grad_ptr1 + (x0), sum_grad_output1, xmask)
    tl.store(output_grad_ptr2 + (x0), sum_grad_output2, xmask)