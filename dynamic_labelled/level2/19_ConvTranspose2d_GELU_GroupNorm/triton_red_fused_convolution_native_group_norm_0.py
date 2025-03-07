# From: 19_ConvTranspose2d_GELU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_native_group_norm_0(
    in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    rnumel = 34848
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x4 = x_index
    x0 = (x_index % 8)
    
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    m2_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r5 = r_index
        r3 = r_index // 4356
        
        input_value = tl.load(in_out_ptr0 + (r5 + 34848 * x4), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        weight_value = tl.load(in_ptr0 + (r3 + 8 * x0), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        
        fused_value = input_value + weight_value
        
        half = 0.5
        scaled_value = fused_value * half
        sqrt_half = 0.7071067811865476
        erf_input = fused_value * sqrt_half
        erf_result = tl.extra.cuda.libdevice.erf(erf_input)
        
        adjusted_value = scaled_value * (erf_result + 1.0)
        broadcasted_value = tl.broadcast_to(adjusted_value, [XBLOCK, RBLOCK])
        
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            broadcasted_value, mean_accumulator, m2_accumulator, weight_accumulator, r_offset == 0
        )
        
        mean_accumulator = tl.where(r_mask & x_mask, mean_next, mean_accumulator)
        m2_accumulator = tl.where(r_mask & x_mask, m2_next, m2_accumulator)
        weight_accumulator = tl.where(r_mask & x_mask, weight_next, weight_accumulator)
        
        tl.store(in_out_ptr0 + (r5 + 34848 * x4), fused_value, r_mask & x_mask)
    
    mean_result, m2_result, weight_result = triton_helpers.welford(
        mean_accumulator, m2_accumulator, weight_accumulator, 1
    )
    
    mean_result = mean_result[:, None]
    m2_result = m2_result[:, None]
    
    tl.store(out_ptr0 + (x4), mean_result, x_mask)
    tl.store(out_ptr1 + (x4), m2_result, x_mask)
    
    num_elements = 34848.0
    variance = m2_result / num_elements
    epsilon = 1e-05
    adjusted_variance = variance + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    
    tl.store(out_ptr2 + (x4), inv_stddev, x_mask)