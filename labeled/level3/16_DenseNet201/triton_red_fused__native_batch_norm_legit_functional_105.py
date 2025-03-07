# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_105(
    input_ptr_mean, input_ptr_var, input_ptr_input, 
    output_ptr_mean, output_ptr_var, output_ptr_output, 
    output_ptr_normalized, output_ptr_scaled, 
    total_elements, reduction_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    total_elements = 448
    reduction_elements = 1960
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    
    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_elements
        r_channel = (r_indices % 196)
        r_feature_map = r_indices // 196
        input_values = tl.load(
            input_ptr_mean + (r_channel + 196 * x_indices_flat + 87808 * r_feature_map), 
            r_mask & x_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        broadcasted_values = tl.broadcast_to(input_values, [XBLOCK, RBLOCK])
        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_reduce(
            broadcasted_values, running_mean, running_m2, running_weight, r_offset == 0
        )
        running_mean = tl.where(r_mask & x_mask, running_mean_next, running_mean)
        running_m2 = tl.where(r_mask & x_mask, running_m2_next, running_m2)
        running_weight = tl.where(r_mask & x_mask, running_weight_next, running_weight)
    
    final_mean, final_var, final_weight = triton_helpers.welford(
        running_mean, running_m2, running_weight, 1
    )
    final_mean = final_mean[:, None]
    final_var = final_var[:, None]
    tl.store(output_ptr_mean + (x_indices_flat), final_mean, x_mask)
    tl.store(output_ptr_var + (x_indices_flat), final_var, x_mask)
    
    input_mean = tl.load(input_ptr_input + (x_indices_flat), x_mask, eviction_policy='evict_last')
    input_var = tl.load(input_ptr_var + (x_indices_flat), x_mask, eviction_policy='evict_last')
    
    num_reduction_elements = 1960.0
    epsilon = 1e-05
    inv_std = (final_var / num_reduction_elements + epsilon).rsqrt()
    
    momentum = 0.1
    running_mean_scaled = final_mean * momentum
    input_mean_scaled = input_mean * 0.9
    adjusted_mean = running_mean_scaled + input_mean_scaled
    
    variance_scale = 1.0005104645227156
    variance_scaled = final_var * variance_scale * momentum
    adjusted_var = variance_scaled + input_var * 0.9
    
    tl.store(output_ptr_output + (x_indices_flat), inv_std, x_mask)
    tl.store(output_ptr_normalized + (x_indices_flat), adjusted_mean, x_mask)
    tl.store(output_ptr_scaled + (x_indices_flat), adjusted_var, x_mask)