# From: 23_EfficientNetB1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_29(
    input_ptr_mean, input_ptr_var, input_ptr_gamma, input_ptr_beta, input_ptr_running_var,
    output_ptr_mean, output_ptr_inv_std, output_ptr_running_mean, output_ptr_running_var,
    total_elements, running_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    total_elements = 144
    running_elements = 71
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    running_mean_accum = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_var_accum = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight_accum = tl.zeros([XBLOCK, RBLOCK], tl.float32)

    for r_offset in range(0, running_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < running_elements
        r_indices_flat = r_indices
        input_mean = tl.load(input_ptr_mean + (x_indices_flat + 144 * r_indices_flat), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        input_var = tl.load(input_ptr_var + (x_indices_flat + 144 * r_indices_flat), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        input_gamma = tl.load(input_ptr_gamma + (x_indices_flat + 144 * r_indices_flat), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        
        broadcast_mean = tl.broadcast_to(input_mean, [XBLOCK, RBLOCK])
        broadcast_var = tl.broadcast_to(input_var, [XBLOCK, RBLOCK])
        broadcast_gamma = tl.broadcast_to(input_gamma, [XBLOCK, RBLOCK])
        
        running_mean_next, running_var_next, running_weight_next = triton_helpers.welford_combine(
            running_mean_accum, running_var_accum, running_weight_accum,
            broadcast_mean, broadcast_var, broadcast_gamma
        )
        
        running_mean_accum = tl.where(r_mask & x_mask, running_mean_next, running_mean_accum)
        running_var_accum = tl.where(r_mask & x_mask, running_var_next, running_var_accum)
        running_weight_accum = tl.where(r_mask & x_mask, running_weight_next, running_weight_accum)

    mean, variance, weight = triton_helpers.welford(
        running_mean_accum, running_var_accum, running_weight_accum, 1
    )
    
    mean_broadcast = mean[:, None]
    inv_std_broadcast = (variance / 9000.0 + 1e-05).rsqrt()
    weight_broadcast = weight[:, None]
    
    tl.store(output_ptr_mean + (x_indices_flat), mean_broadcast, x_mask)
    
    running_mean = tl.load(input_ptr_running_mean + (x_indices_flat), x_mask, eviction_policy='evict_last')
    running_var = tl.load(input_ptr_running_var + (x_indices_flat), x_mask, eviction_policy='evict_last')
    
    running_mean_updated = mean_broadcast * 0.1 + running_mean * 0.9
    running_var_updated = (variance / 9000.0 * 1.000111123458162 * 0.1) + running_var * 0.9
    
    tl.store(output_ptr_inv_std + (x_indices_flat), inv_std_broadcast, x_mask)
    tl.store(output_ptr_running_mean + (x_indices_flat), running_mean_updated, x_mask)
    tl.store(output_ptr_running_var + (x_indices_flat), running_var_updated, x_mask)