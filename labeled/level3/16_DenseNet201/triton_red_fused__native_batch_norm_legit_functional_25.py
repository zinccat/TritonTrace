# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_25(
    input_ptr_mean, input_ptr_var, input_ptr_input, 
    output_ptr_mean, output_ptr_var, output_ptr_output, 
    output_ptr_normalized, output_ptr_scaled, 
    total_elements, reduction_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    total_elements = 224
    reduction_elements = 31360
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
        r_index_mod = (r_indices % 3136)
        r_index_div = r_indices // 3136
        input_data = tl.load(
            input_ptr_input + (r_index_mod + 3136 * x_indices_flat + 702464 * r_index_div), 
            r_mask & x_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        broadcast_input = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_reduce(
            broadcast_input, running_mean, running_m2, running_weight, r_offset == 0
        )
        running_mean = tl.where(r_mask & x_mask, running_mean_next, running_mean)
        running_m2 = tl.where(r_mask & x_mask, running_m2_next, running_m2)
        running_weight = tl.where(r_mask & x_mask, running_weight_next, running_weight)
    
    final_mean, final_var, final_weight = triton_helpers.welford(
        running_mean, running_m2, running_weight, 1
    )
    final_mean_broadcast = final_mean[:, None]
    final_var_broadcast = final_var[:, None]
    final_weight_broadcast = final_weight[:, None]
    
    tl.store(output_ptr_mean + (x_indices_flat), final_mean_broadcast, x_mask)
    tl.store(output_ptr_var + (x_indices_flat), final_var_broadcast, x_mask)
    
    input_mean = tl.load(input_ptr_mean + (x_indices_flat), x_mask, eviction_policy='evict_last')
    input_var = tl.load(input_ptr_var + (x_indices_flat), x_mask, eviction_policy='evict_last')
    
    num_reduction_elements = 31360.0
    mean_div_num_elements = final_var_broadcast / num_reduction_elements
    epsilon = 1e-05
    mean_var_sum = mean_div_num_elements + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(mean_var_sum)
    
    momentum = 0.1
    running_mean_scaled = final_mean_broadcast * momentum
    momentum_factor = 0.9
    input_mean_scaled = input_mean * momentum_factor
    updated_mean = running_mean_scaled + input_mean_scaled
    
    variance_factor = 1.0000318887719635
    mean_variance_scaled = mean_div_num_elements * variance_factor
    variance_scaled = mean_variance_scaled * momentum
    updated_var = variance_scaled + input_var * momentum_factor
    
    tl.store(output_ptr_normalized + (x_indices_flat), inv_std, x_mask)
    tl.store(output_ptr_scaled + (x_indices_flat), updated_mean, x_mask)
    tl.store(output_ptr_output + (x_indices_flat), updated_var, x_mask)