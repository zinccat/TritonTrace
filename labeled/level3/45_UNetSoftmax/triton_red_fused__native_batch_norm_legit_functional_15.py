# From: 45_UNetSoftmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_15(
    input_ptr_mean, input_ptr_var, input_ptr_input, 
    output_ptr_mean, output_ptr_var, output_ptr_normalized, 
    output_ptr_scaled, output_ptr_shifted, 
    num_elements, num_reduction_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements = 512
    num_reduction_elements = 4096
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    
    for r_offset in range(0, num_reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < num_reduction_elements
        r1 = (r_indices % 512)
        r2 = r_indices // 512
        input_data = tl.load(
            input_ptr_input + (r1 + 512 * x_indices_flat + 262144 * r2), 
            rmask & x_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        broadcast_input = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_reduce(
            broadcast_input, running_mean, running_m2, running_weight, r_offset == 0
        )
        running_mean = tl.where(rmask & x_mask, running_mean_next, running_mean)
        running_m2 = tl.where(rmask & x_mask, running_m2_next, running_m2)
        running_weight = tl.where(rmask & x_mask, running_weight_next, running_weight)
    
    final_mean, final_var, final_weight = triton_helpers.welford(
        running_mean, running_m2, running_weight, 1
    )
    final_mean = final_mean[:, None]
    final_var = final_var[:, None]
    tl.store(output_ptr_mean + (x_indices_flat), final_mean, x_mask)
    tl.store(output_ptr_var + (x_indices_flat), final_var, x_mask)
    
    input_mean = tl.load(input_ptr_mean + (x_indices_flat), x_mask, eviction_policy='evict_last')
    input_var = tl.load(input_ptr_var + (x_indices_flat), x_mask, eviction_policy='evict_last')
    
    epsilon = 1e-05
    inv_std = tl.extra.cuda.libdevice.rsqrt(final_var + epsilon)
    momentum = 0.1
    adjusted_mean = final_mean * momentum
    running_mean = adjusted_mean + input_mean * (1 - momentum)
    
    variance_scale = 0.9
    adjusted_var = final_var * variance_scale
    running_var = adjusted_var + input_var * (1 - variance_scale)
    
    variance_correction = 1.0002442002442002
    corrected_var = final_var * variance_correction
    scaled_mean = corrected_var * momentum
    
    tl.store(output_ptr_normalized + (x_indices_flat), inv_std, x_mask)
    tl.store(output_ptr_scaled + (x_indices_flat), running_mean, x_mask)
    tl.store(output_ptr_shifted + (x_indices_flat), running_var, x_mask)