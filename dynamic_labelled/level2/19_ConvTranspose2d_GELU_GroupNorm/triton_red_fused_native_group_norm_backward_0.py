# From: 19_ConvTranspose2d_GELU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_0red_fused_native_group_norm_backward_0(
    input_grad_ptr, input_ptr, output_grad_ptr0, output_grad_ptr1, xnumel, rnumel, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    rnumel = 4356
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x_indices = xindex
    sum_grad_x = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    sum_input_x = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r_indices = rindex
        grad_input = tl.load(input_grad_ptr + (r_indices + 4356 * x_indices), rmask & xmask, eviction_policy='evict_first', other=0.0)
        input_data = tl.load(input_ptr + (r_indices + 4356 * x_indices), rmask & xmask, eviction_policy='evict_first', other=0.0)
        
        half = 0.5
        sqrt_inv_2 = 0.7071067811865476
        erf_input = input_data * sqrt_inv_2
        erf_result = tl.extra.cuda.libdevice.erf(erf_input)
        one = 1.0
        gelu_output = (input_data * half) * (erf_result + one)
        grad_output = grad_input * gelu_output
        
        grad_output_broadcast = tl.broadcast_to(grad_output, [XBLOCK, RBLOCK])
        sum_grad_x = tl.where(rmask & xmask, sum_grad_x + grad_output_broadcast, sum_grad_x)
        
        input_data_broadcast = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        sum_input_x = tl.where(rmask & xmask, sum_input_x + input_data_broadcast, sum_input_x)

    sum_grad_x_reduced = tl.sum(sum_grad_x, 1)[:, None]
    sum_input_x_reduced = tl.sum(sum_input_x, 1)[:, None]
    
    tl.store(output_grad_ptr0 + (x_indices), sum_grad_x_reduced, xmask)
    tl.store(output_grad_ptr1 + (x_indices), sum_input_x_reduced, xmask)