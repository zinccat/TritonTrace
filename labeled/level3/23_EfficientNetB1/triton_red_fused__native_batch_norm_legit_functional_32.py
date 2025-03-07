# From: 23_EfficientNetB1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_32red_fused__native_batch_norm_legit_functional_32(
    input_ptr_mean, input_ptr_var, input_ptr_count, input_ptr_running_mean, input_ptr_running_var,
    output_ptr_mean, output_ptr_var, output_ptr_running_mean, output_ptr_running_var, output_ptr_count,
    total_elements, reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    total_elements = 40
    reduction_elements = 71
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_count = tl.zeros([XBLOCK, RBLOCK], tl.float32)

    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_elements
        r_indices_flat = r_indices
        input_mean = tl.load(input_ptr_mean + (x_indices_flat + 40 * r_indices_flat), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        input_var = tl.load(input_ptr_var + (x_indices_flat + 40 * r_indices_flat), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        input_count = tl.load(input_ptr_count + (x_indices_flat + 40 * r_indices_flat), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        
        broadcast_mean = tl.broadcast_to(input_mean, [XBLOCK, RBLOCK])
        broadcast_var = tl.broadcast_to(input_var, [XBLOCK, RBLOCK])
        broadcast_count = tl.broadcast_to(input_count, [XBLOCK, RBLOCK])
        
        running_mean_next, running_m2_next, running_count_next = triton_helpers.welford_combine(
            running_mean, running_m2, running_count,
            broadcast_mean, broadcast_var, broadcast_count
        )
        
        running_mean = tl.where(r_mask & x_mask, running_mean_next, running_mean)
        running_m2 = tl.where(r_mask & x_mask, running_m2_next, running_m2)
        running_count = tl.where(r_mask & x_mask, running_count_next, running_count)

    final_mean, final_var, final_count = triton_helpers.welford(
        running_mean, running_m2, running_count, 1
    )
    
    final_mean = final_mean[:, None]
    final_var = final_var[:, None]
    final_count = final_count[:, None]
    
    tl.store(output_ptr_mean + (x_indices_flat), final_mean, x_mask)
    tl.store(output_ptr_var + (x_indices_flat), final_var, x_mask)
    
    running_mean_input = tl.load(input_ptr_running_mean + (x_indices_flat), x_mask, eviction_policy='evict_last')
    running_var_input = tl.load(input_ptr_running_var + (x_indices_flat), x_mask, eviction_policy='evict_last')
    
    epsilon = 9000.0
    var_epsilon = 1e-05
    inv_std = tl.extra.cuda.libdevice.rsqrt(final_var / epsilon + var_epsilon)
    
    momentum = 0.1
    adjusted_mean = final_mean * momentum
    decay = 0.9
    updated_running_mean = running_mean_input * decay + adjusted_mean
    
    scale_factor = 1.000111123458162
    adjusted_var = final_var * scale_factor * momentum
    updated_running_var = running_var_input * decay + adjusted_var
    
    tl.store(output_ptr_running_mean + (x_indices_flat), inv_std, x_mask)
    tl.store(output_ptr_running_mean + (x_indices_flat), updated_running_mean, x_mask)
    tl.store(output_ptr_running_var + (x_indices_flat), updated_running_var, x_mask)