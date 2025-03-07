# From: 92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused_convolution_native_group_norm_0(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xnumel = 1024
    rnumel = 1800
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x4 = x_index
    x0 = x_index % 8
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    m2_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r5 = r_index
        r3 = (r_index // 900)
        
        input_data = tl.load(in_out_ptr0 + (r5 + (1800 * x4)), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        weight_data = tl.load(in_ptr0 + (r3 + (2 * x0)), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        combined_data = input_data + weight_data
        broadcast_data = tl.broadcast_to(combined_data, [XBLOCK, RBLOCK])
        
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            broadcast_data, mean_accumulator, m2_accumulator, weight_accumulator, r_offset == 0
        )
        
        mean_accumulator = tl.where(r_mask & x_mask, mean_next, mean_accumulator)
        m2_accumulator = tl.where(r_mask & x_mask, m2_next, m2_accumulator)
        weight_accumulator = tl.where(r_mask & x_mask, weight_next, weight_accumulator)
        
        tl.store(in_out_ptr0 + (r5 + (1800 * x4)), combined_data, r_mask & x_mask)
    
    mean_final, variance_final, weight_final = triton_helpers.welford(
        mean_accumulator, m2_accumulator, weight_accumulator, 1
    )
    
    mean_final = mean_final[:, None]
    variance_final = variance_final[:, None]
    weight_final = weight_final[:, None]
    
    tl.store(out_ptr0 + (x4), mean_final, x_mask)
    
    normalization_factor = 1800.0
    variance_normalized = variance_final / normalization_factor
    epsilon = 1e-05
    variance_adjusted = variance_normalized + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_adjusted)
    
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), inv_sqrt_variance, x_mask)