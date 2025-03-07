# From: 19_ConvTranspose2d_GELU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_0(input_grad_ptr, input_ptr, output_grad_ptr0, output_grad_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    rnumel = 4356
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    temp_sum0 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    temp_sum1 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r_indices_flat = r_indices
        input_grad = tl.load(input_grad_ptr + (r_indices_flat + 4356 * x_indices_flat), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        input = tl.load(input_ptr + (r_indices_flat + 4356 * x_indices_flat), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        
        half = 0.5
        scaled_input = input * half
        sqrt_inv_2 = 0.7071067811865476
        erf_input = input * sqrt_inv_2
        erf_result = tl.extra.cuda.libdevice.erf(erf_input)
        one = 1.0
        erf_adjusted = erf_result + one
        activation_grad = scaled_input * erf_adjusted
        grad_wrt_input = input_grad * activation_grad
        
        temp_sum0 += tl.where(r_mask & x_mask, tl.broadcast_to(grad_wrt_input, [XBLOCK, RBLOCK]), temp_sum0)
        temp_sum1 += tl.where(r_mask & x_mask, tl.broadcast_to(input_grad, [XBLOCK, RBLOCK]), temp_sum1)
    
    output_grad0 = tl.sum(temp_sum0, 1)[:, None]
    output_grad1 = tl.sum(temp_sum1, 1)[:, None]
    
    tl.store(output_grad_ptr0 + (x_indices_flat), output_grad0, x_mask)
    tl.store(output_grad_ptr1 + (x_indices_flat), output_grad1, x_mask)