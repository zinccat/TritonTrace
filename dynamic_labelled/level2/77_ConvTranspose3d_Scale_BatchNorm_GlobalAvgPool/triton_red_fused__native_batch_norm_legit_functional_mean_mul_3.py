# From: 77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_mean_mul_3(
    output_ptr, input_ptr, mean_ptr, variance_ptr, gamma_ptr, beta_ptr, kernel_size_0, kernel_size_1, 
    x_num_elements, r_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):

    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x3 = x_index
    x0 = (x_index % 32)
    
    mean_value = tl.load(mean_ptr + (x0), x_mask, eviction_policy='evict_last')
    variance_value = tl.load(variance_ptr + (x0), x_mask, eviction_policy='evict_last')
    gamma_value = tl.load(gamma_ptr + (x0), x_mask, eviction_policy='evict_last')
    beta_value = tl.load(beta_ptr + (x0), x_mask, eviction_policy='evict_last')
    
    accumulated_result = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r2 = r_index
        
        input_value = tl.load(
            input_ptr + (r2 + 8*x3 + 2*x3*kernel_size_1*kernel_size_1 + 4*kernel_size_0*x3 + 8*kernel_size_1*x3 + 
                         kernel_size_0*x3*kernel_size_1*kernel_size_1 + 4*kernel_size_0*kernel_size_1*x3), 
            rmask & x_mask, eviction_policy='evict_first', other=0.0
        )
        
        scale_factor = 2.0
        scaled_input = input_value * scale_factor
        centered_input = scaled_input - mean_value
        
        normalization_factor = (4*kernel_size_0*kernel_size_0 + 8*kernel_size_0 + kernel_size_0*kernel_size_0*kernel_size_1*kernel_size_1 + 
                                2*kernel_size_0*kernel_size_1*kernel_size_1 + 4*kernel_size_1*kernel_size_0*kernel_size_0 + 
                                8*kernel_size_0*kernel_size_1).to(tl.float32)
        
        variance_normalized = variance_value / normalization_factor
        epsilon = 1e-05
        variance_stabilized = variance_normalized + epsilon
        rsqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_stabilized)
        
        normalized_input = centered_input * rsqrt_variance
        scaled_normalized_input = normalized_input * gamma_value
        shifted_scaled_input = scaled_normalized_input + beta_value
        
        broadcasted_shifted_scaled_input = tl.broadcast_to(shifted_scaled_input, [XBLOCK, RBLOCK])
        accumulated_result = accumulated_result + broadcasted_shifted_scaled_input
        
        accumulated_result = tl.where(rmask & x_mask, accumulated_result, accumulated_result)
    
    summed_result = tl.sum(accumulated_result, 1)[:, None]
    normalization_constant = (8 + 2*kernel_size_1*kernel_size_1 + 4*kernel_size_0 + 8*kernel_size_1 + 
                              kernel_size_0*kernel_size_1*kernel_size_1 + 4*kernel_size_0*kernel_size_1).to(tl.float32)
    
    final_result = summed_result / normalization_constant
    
    tl.debug_barrier()
    tl.store(output_ptr + (x3), final_result, x_mask)