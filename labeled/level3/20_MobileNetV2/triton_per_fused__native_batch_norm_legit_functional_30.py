# From: 20_MobileNetV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_30(
    input_ptr_mean, input_ptr_var, input_ptr_count, input_ptr_running_mean, input_ptr_running_var,
    output_ptr_normalized, output_ptr_running_mean, output_ptr_running_var, output_ptr_mean, output_ptr_var, 
    num_elements, running_num_elements, XBLOCK: tl.constexpr
):
    num_elements = 32
    running_num_elements = 62
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < running_num_elements
    r_indices_broadcast = r_indices
    x_indices_broadcast = x_indices
    loaded_mean = tl.load(input_ptr_mean + (x_indices_broadcast + 32 * r_indices_broadcast), r_mask & x_mask, other=0.0)
    loaded_var = tl.load(input_ptr_var + (x_indices_broadcast + 32 * r_indices_broadcast), r_mask & x_mask, other=0.0)
    loaded_count = tl.load(input_ptr_count + (x_indices_broadcast + 32 * r_indices_broadcast), r_mask & x_mask, other=0.0)
    running_mean = tl.load(input_ptr_running_mean + (x_indices_broadcast), x_mask, eviction_policy='evict_last')
    running_var = tl.load(input_ptr_running_var + (x_indices_broadcast), x_mask, eviction_policy='evict_last')
    
    broadcast_mean = tl.broadcast_to(loaded_mean, [XBLOCK, RBLOCK])
    broadcast_var = tl.broadcast_to(loaded_var, [XBLOCK, RBLOCK])
    broadcast_count = tl.broadcast_to(loaded_count, [XBLOCK, RBLOCK])
    
    masked_mean = tl.where(r_mask & x_mask, broadcast_mean, 0)
    masked_var = tl.where(r_mask & x_mask, broadcast_var, 0)
    masked_count = tl.where(r_mask & x_mask, broadcast_count, 0)
    
    mean, var, count = triton_helpers.welford(masked_mean, masked_var, masked_count, 1)
    
    mean_broadcast = mean[:, None]
    var_broadcast = var[:, None]
    
    normalization_factor = 7840.0
    epsilon = 1e-05
    adjusted_var = var_broadcast / normalization_factor
    adjusted_var_with_epsilon = adjusted_var + epsilon
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(adjusted_var_with_epsilon)
    
    momentum = 0.1
    scaled_mean = mean * momentum
    decay = 0.9
    updated_running_mean = scaled_mean + running_mean * decay
    
    bias_correction = 1.0001275672917465
    corrected_var = adjusted_var * bias_correction
    scaled_corrected_var = corrected_var * momentum
    updated_running_var = scaled_corrected_var + running_var * decay
    
    tl.store(output_ptr_normalized + (x_indices_broadcast), reciprocal_sqrt, x_mask)
    tl.store(output_ptr_running_mean + (x_indices_broadcast), updated_running_mean, x_mask)
    tl.store(output_ptr_running_var + (x_indices_broadcast), updated_running_var, x_mask)
    tl.store(output_ptr_mean + (x_indices_broadcast), mean_broadcast, x_mask)
    tl.store(output_ptr_var + (x_indices_broadcast), var_broadcast, x_mask)