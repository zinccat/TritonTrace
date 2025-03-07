# From: 31_VisionAttention

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_7(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, output_ptr1, output_ptr2, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    rnumel = 128
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_base = tl.arange(0, RBLOCK)[None, :]
    x3 = x_index
    x0 = (x_index % 2)
    x1 = x_index // 2
    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r2 = r_index
        input_val0 = tl.load(input_ptr0 + (r2 + 128 * x3), r_mask, eviction_policy='evict_first', other=0.0)
        input_val1 = tl.load(input_ptr1 + (r2), r_mask, eviction_policy='evict_last', other=0.0)
        input_val2 = tl.load(input_ptr2 + (x1 + 16384 * r2 + 2097152 * x0), r_mask, eviction_policy='evict_last', other=0.0)
        
        sum_val0_1 = input_val0 + input_val1
        sum_val0_1_2 = sum_val0_1 + input_val2
        broadcast_sum = tl.broadcast_to(sum_val0_1_2, [XBLOCK, RBLOCK])
        
        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_reduce(
            broadcast_sum, running_mean, running_m2, running_weight, r_offset == 0
        )
        
        running_mean = tl.where(r_mask, running_mean_next, running_mean)
        running_m2 = tl.where(r_mask, running_m2_next, running_m2)
        running_weight = tl.where(r_mask, running_weight_next, running_weight)
    
    mean, variance, weight = triton_helpers.welford(running_mean, running_m2, running_weight, 1)
    mean = mean[:, None]
    variance = variance[:, None]
    
    tl.store(output_ptr0 + (x3), mean, None)
    tl.store(output_ptr1 + (x3), variance, None)
    
    scale_factor = 128.0
    normalized_variance = variance / scale_factor
    epsilon = 1e-05
    adjusted_variance = normalized_variance + epsilon
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    gamma = 0.0078125
    normalized_gamma = reciprocal_sqrt * gamma
    
    tl.store(output_ptr2 + (x3), normalized_gamma, None)