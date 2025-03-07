# From: 84_Gemm_BatchNorm_Scaling_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__softmax_backward_data_mul_native_batch_norm_backward_2(
    input_grad_ptr, input_data_ptr, scale_ptr, scale_broadcast_ptr, input_grad_accum_ptr, input_data_accum_ptr, output_ptr, 
    out_grad_ptr, out_grad_scaled_ptr, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):

    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x_indices = xindex
    scale_value = tl.load(scale_broadcast_ptr + (0))
    scale_broadcast = tl.broadcast_to(scale_value, [XBLOCK, RBLOCK])
    grad_accum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_grad = tl.load(input_grad_accum_ptr + (x_indices), xmask, eviction_policy='evict_last')
    grad_accum_temp = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r_indices = rindex
        input_grad_data = tl.load(input_data_ptr + (x_indices + 512 * r_indices), rmask & xmask, eviction_policy='evict_first', other=0.0)
        scale = tl.load(scale_ptr + (r_indices), rmask, eviction_policy='evict_last', other=0.0)
        input_data = tl.load(input_data_accum_ptr + (x_indices + 512 * r_indices), rmask & xmask, eviction_policy='evict_first', other=0.0)
        input_grad_accum_data = tl.load(input_data_accum_ptr + (x_indices + 512 * r_indices), rmask & xmask, eviction_policy='evict_first', other=0.0)
        
        neg_input_grad_data = -input_grad_data
        scaled_input_data = input_data * input_grad_data
        fused_multiply_add = tl.extra.cuda.libdevice.fma(neg_input_grad_data, scale, scaled_input_data)
        scaled_fma = fused_multiply_add * scale_broadcast
        scaled_fma_broadcast = tl.broadcast_to(scaled_fma, [XBLOCK, RBLOCK])
        grad_accum_update = grad_accum + scaled_fma_broadcast
        grad_accum = tl.where(rmask & xmask, grad_accum_update, grad_accum)
        
        input_grad_diff = input_grad_accum_data - input_grad
        scaled_grad_diff = scaled_fma * input_grad_diff
        scaled_grad_diff_broadcast = tl.broadcast_to(scaled_grad_diff, [XBLOCK, RBLOCK])
        grad_accum_temp_update = grad_accum_temp + scaled_grad_diff_broadcast
        grad_accum_temp = tl.where(rmask & xmask, grad_accum_temp_update, grad_accum_temp)

    grad_accum_sum = tl.sum(grad_accum, 1)[:, None]
    grad_accum_temp_sum = tl.sum(grad_accum_temp, 1)[:, None]
    tl.store(out_grad_ptr + (x_indices), grad_accum_sum, xmask)
    tl.store(out_grad_scaled_ptr + (x_indices), grad_accum_temp_sum, xmask)
    
    output_grad = tl.load(output_ptr + (x_indices), xmask, eviction_policy='evict_last')
    scaled_output_grad = grad_accum_temp_sum * output_grad
    tl.store(out_grad_scaled_ptr + (x_indices), scaled_output_grad, xmask)