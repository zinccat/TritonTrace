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
    r_mask_full = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = x_index
    sum_grad_x = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    sum_x = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r1 = r_index
        grad_x = tl.load(input_grad_ptr + (x0 + 4194304 * r1), r_mask, eviction_policy='evict_first', other=0.0)
        x = tl.load(input_grad_ptr + (x0 + 4194304 * r1), r_mask, eviction_policy='evict_first', other=0.0)
        mean = tl.load(mean_ptr + (r1), r_mask, eviction_policy='evict_last', other=0.0)
        inv_std = tl.load(inv_std_ptr + (r1), r_mask, eviction_policy='evict_last', other=0.0)
        
        x_centered = x - mean
        scaled_x_centered = x_centered * inv_std
        grad_x_scaled = grad_x * scaled_x_centered
        grad_x_broadcast = tl.broadcast_to(grad_x_scaled, [XBLOCK, RBLOCK])
        
        sum_grad_x += tl.where(r_mask, grad_x_broadcast, sum_grad_x)
        x_broadcast = tl.broadcast_to(x, [XBLOCK, RBLOCK])
        sum_x += tl.where(r_mask, x_broadcast, sum_x)

    output_grad_mean = tl.sum(sum_grad_x, 1)[:, None]
    output_grad_var = tl.sum(sum_x, 1)[:, None]

    tl.store(output_grad_mean_ptr + (x0), output_grad_mean, None)
    tl.store(output_grad_var_ptr + (x0), output_grad_var, None)