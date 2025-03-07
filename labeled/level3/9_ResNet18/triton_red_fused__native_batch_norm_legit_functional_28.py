# From: 9_ResNet18

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_28red_fused__native_batch_norm_legit_functional_28(
    input_ptr_mean, input_ptr_var, input_ptr_input, 
    output_ptr_mean, output_ptr_var, output_ptr_input, 
    output_ptr_normalized, output_ptr_scale, 
    num_elements, num_features, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements = 512
    num_features = 98
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    
    for r_offset in range(0, num_features, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < num_features
        r_indices_flat = r_indices
        input_data = tl.load(input_ptr_input + (x_indices_flat + 512 * r_indices_flat), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        broadcast_input = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_reduce(
            broadcast_input, running_mean, running_m2, running_weight, r_offset == 0
        )
        running_mean = tl.where(r_mask & x_mask, running_mean_next, running_mean)
        running_m2 = tl.where(r_mask & x_mask, running_m2_next, running_m2)
        running_weight = tl.where(r_mask & x_mask, running_weight_next, running_weight)
    
    mean, variance, weight = triton_helpers.welford(running_mean, running_m2, running_weight, 1)
    mean_broadcast = mean[:, None]
    variance_broadcast = variance[:, None]
    weight_broadcast = weight[:, None]
    
    tl.store(output_ptr_mean + (x_indices_flat), mean_broadcast, x_mask)
    tl.store(output_ptr_var + (x_indices_flat), variance_broadcast, x_mask)
    
    input_data = tl.load(input_ptr_input + (x_indices_flat), x_mask, eviction_policy='evict_last')
    scale_data = tl.load(input_ptr_var + (x_indices_flat), x_mask, eviction_policy='evict_last')
    
    num_features_float = 98.0
    variance_normalized = variance / num_features_float
    epsilon = 1e-05
    variance_normalized_eps = variance_normalized + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(variance_normalized_eps)
    
    momentum = 0.1
    running_mean_scaled = mean * momentum
    momentum_factor = 0.9
    scale_data_momentum = scale_data * momentum_factor
    updated_mean = running_mean_scaled + scale_data_momentum
    
    affine_factor = 1.0103092783505154
    variance_scaled = variance * affine_factor
    variance_scaled_momentum = variance_scaled * momentum
    updated_scale = variance_scaled_momentum + scale_data_momentum
    
    tl.store(output_ptr_normalized + (x_indices_flat), inv_std, x_mask)
    tl.store(output_ptr_scale + (x_indices_flat), updated_mean, x_mask)
    tl.store(output_ptr_input + (x_indices_flat), updated_scale, x_mask)