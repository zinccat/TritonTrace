# From: 14_DenseNet121DenseBlock

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_20(
    input_ptr_mean, input_ptr_var, input_ptr_input, 
    output_ptr_mean, output_ptr_var, output_ptr_input, 
    output_ptr_normalized, output_ptr_scale, 
    num_elements, num_reduction_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements = 192
    num_reduction_elements = 501760
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
        r_index_mod = (r_indices % 50176)
        r_index_div = r_indices // 50176
        input_data = tl.load(
            input_ptr_input + (r_index_mod + 50176 * x_indices_flat + 9633792 * r_index_div), 
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
    
    input_scale = tl.load(input_ptr_scale + (x_indices_flat), x_mask, eviction_policy='evict_last')
    input_bias = tl.load(input_ptr_var + (x_indices_flat), x_mask, eviction_policy='evict_last')
    
    num_reduction_elements_float = 501760.0
    variance_adjusted = final_var_broadcast / num_reduction_elements_float
    epsilon = 1e-05
    variance_adjusted_epsilon = variance_adjusted + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_adjusted_epsilon)
    
    scale_factor = 1.0000019929886659
    variance_scaled = variance_adjusted * scale_factor
    momentum = 0.1
    variance_scaled_momentum = variance_scaled * momentum
    momentum_input_scale = 0.9
    input_scale_momentum = input_scale * momentum_input_scale
    running_scale = variance_scaled_momentum + input_scale_momentum
    
    bias_factor = 0.1
    running_bias = final_mean_broadcast * bias_factor
    input_bias_momentum = input_bias * momentum_input_scale
    running_bias_momentum = running_bias + input_bias_momentum
    
    tl.store(output_ptr_normalized + (x_indices_flat), inv_stddev, x_mask)
    tl.store(output_ptr_scale + (x_indices_flat), running_scale, x_mask)
    tl.store(output_ptr_input + (x_indices_flat), running_bias_momentum, x_mask)