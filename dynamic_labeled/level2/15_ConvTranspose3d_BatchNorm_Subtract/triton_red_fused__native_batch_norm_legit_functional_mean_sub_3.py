# From: 15_ConvTranspose3d_BatchNorm_Subtract

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_mean_sub_3(
    output_ptr, input_ptr, mean_ptr, variance_ptr, gamma_ptr, beta_ptr, kernel_size, 
    x_num_elements, r_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):

    x_num_elements = 512
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x3 = x_index
    x0 = (x_index % 32)
    
    mean = tl.load(mean_ptr + (x0), x_mask, eviction_policy='evict_last')
    variance = tl.load(variance_ptr + (x0), x_mask, eviction_policy='evict_last')
    gamma = tl.load(gamma_ptr + (x0), x_mask, eviction_policy='evict_last')
    beta = tl.load(beta_ptr + (x0), x_mask, eviction_policy='evict_last')
    
    accumulated_output = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r2 = r_index
        input_value = tl.load(
            input_ptr + (r2 + 31 * x3 + ((-124) * kernel_size * x3) + 124 * x3 * kernel_size * kernel_size), 
            r_mask & x_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        
        normalized_input = input_value - mean
        variance_adjustment = 496 + ((-1984) * kernel_size) + 1984 * kernel_size * kernel_size
        variance_adjustment_float = variance_adjustment.to(tl.float32)
        variance_scaled = variance / variance_adjustment_float
        epsilon = 1e-05
        variance_stabilized = variance_scaled + epsilon
        rsqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_stabilized)
        
        normalized_scaled = normalized_input * rsqrt_variance
        scaled_output = normalized_scaled * gamma
        output_with_bias = scaled_output + beta
        
        broadcast_output = tl.broadcast_to(output_with_bias, [XBLOCK, RBLOCK])
        accumulated_output = accumulated_output + broadcast_output
        accumulated_output = tl.where(r_mask & x_mask, accumulated_output, accumulated_output)
        
        tl.store(
            output_ptr + (r2 + 31 * x3 + ((-124) * kernel_size * x3) + 124 * x3 * kernel_size * kernel_size), 
            output_with_bias, 
            r_mask & x_mask
        )
    
    mean_accumulated = tl.sum(accumulated_output, 1)[:, None]
    
    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r2 = r_index
        stored_output = tl.load(
            output_ptr + (r2 + 31 * x3 + ((-124) * kernel_size * x3) + 124 * x3 * kernel_size * kernel_size), 
            r_mask & x_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        
        mean_adjustment = 31 + ((-124) * kernel_size) + 124 * kernel_size * kernel_size
        mean_adjustment_float = mean_adjustment.to(tl.float32)
        mean_correction = mean_accumulated / mean_adjustment_float
        corrected_output = stored_output - mean_correction
        
        tl.store(
            output_ptr + (r2 + 31 * x3 + ((-124) * kernel_size * x3) + 124 * x3 * kernel_size * kernel_size), 
            corrected_output, 
            r_mask & x_mask
        )