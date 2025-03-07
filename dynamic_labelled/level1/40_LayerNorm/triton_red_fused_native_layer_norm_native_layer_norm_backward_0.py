# From: 40_LayerNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_layer_norm_native_layer_norm_backward_0(
    input_grad_ptr, mean_ptr, variance_ptr, inv_std_ptr, output_grad_mean_ptr, output_grad_var_ptr, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    full_mask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = x_index
    temp_sum_grad = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    temp_sum_input = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r1 = r_index
        grad_input = tl.load(input_grad_ptr + (x0 + 4194304 * r1), r_mask, eviction_policy='evict_first', other=0.0)
        input_data = tl.load(input_grad_ptr + (x0 + 4194304 * r1), r_mask, eviction_policy='evict_first', other=0.0)
        mean = tl.load(mean_ptr + (r1), r_mask, eviction_policy='evict_last', other=0.0)
        inv_std = tl.load(inv_std_ptr + (r1), r_mask, eviction_policy='evict_last', other=0.0)
        
        centered_input = input_data - mean
        scaled_grad = centered_input * inv_std
        grad_scaled_input = grad_input * scaled_grad
        broadcast_grad_scaled_input = tl.broadcast_to(grad_scaled_input, [XBLOCK, RBLOCK])
        
        temp_sum_grad = temp_sum_grad + broadcast_grad_scaled_input
        temp_sum_grad = tl.where(r_mask, temp_sum_grad, temp_sum_grad)
        
        broadcast_input = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        temp_sum_input = temp_sum_input + broadcast_input
        temp_sum_input = tl.where(r_mask, temp_sum_input, temp_sum_input)

    sum_grad = tl.sum(temp_sum_grad, 1)[:, None]
    sum_input = tl.sum(temp_sum_input, 1)[:, None]
    
    tl.store(output_grad_mean_ptr + (x0), sum_grad, None)
    tl.store(output_grad_var_ptr + (x0), sum_input, None)