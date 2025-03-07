# From: 77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_mean_mul_3(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, kernel_size0, kernel_size1, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x3 = x_index
    x0 = (x_index % 32)
    
    mean = tl.load(in_ptr1 + (x0), x_mask, eviction_policy='evict_last')
    variance = tl.load(in_ptr2 + (x0), x_mask, eviction_policy='evict_last')
    gamma = tl.load(in_ptr3 + (x0), x_mask, eviction_policy='evict_last')
    beta = tl.load(in_ptr4 + (x0), x_mask, eviction_policy='evict_last')
    
    result_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r2 = r_index
        
        input_value = tl.load(
            in_ptr0 + (r2 + 8*x3 + 2*x3*kernel_size1*kernel_size1 + 4*kernel_size0*x3 + 8*kernel_size1*x3 + kernel_size0*x3*kernel_size1*kernel_size1 + 4*kernel_size0*kernel_size1*x3),
            r_mask & x_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        
        scale_factor = 2.0
        scaled_input = input_value * scale_factor
        centered_input = scaled_input - mean
        
        normalization_factor = (
            4*kernel_size0*kernel_size0 + 8*kernel_size0 + kernel_size0*kernel_size0*kernel_size1*kernel_size1 +
            2*kernel_size0*kernel_size1*kernel_size1 + 4*kernel_size1*kernel_size0*kernel_size0 + 8*kernel_size0*kernel_size1
        )
        normalization_factor = normalization_factor.to(tl.float32)
        
        variance_adjusted = variance / normalization_factor
        epsilon = 1e-05
        variance_adjusted += epsilon
        inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_adjusted)
        
        normalized_input = centered_input * inv_stddev
        scaled_normalized_input = normalized_input * gamma
        output = scaled_normalized_input + beta
        
        broadcast_output = tl.broadcast_to(output, [XBLOCK, RBLOCK])
        result_accumulator = result_accumulator + broadcast_output
        
        result_accumulator = tl.where(r_mask & x_mask, result_accumulator, result_accumulator)
    
    sum_result = tl.sum(result_accumulator, 1)[:, None]
    normalization_constant = 8 + 2*kernel_size1*kernel_size1 + 4*kernel_size0 + 8*kernel_size1 + kernel_size0*kernel_size1*kernel_size1 + 4*kernel_size0*kernel_size1
    normalization_constant = normalization_constant.to(tl.float32)
    
    final_result = sum_result / normalization_constant
    
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), final_result, x_mask)