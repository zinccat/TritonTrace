# From: 21_EfficientNetMBConv

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_6red_fused__native_batch_norm_legit_functional_6(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, 
    output_ptr0, output_ptr2, output_ptr4, output_ptr6, 
    total_elements, reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    total_elements = 672
    reduction_elements = 112
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
        r_indices_flat = r_indices
        data0 = tl.load(input_ptr0 + (x_indices_flat + 672 * r_indices_flat), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        data1 = tl.load(input_ptr1 + (x_indices_flat + 672 * r_indices_flat), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        data2 = tl.load(input_ptr2 + (x_indices_flat + 672 * r_indices_flat), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        
        broadcast_data0 = tl.broadcast_to(data0, [XBLOCK, RBLOCK])
        broadcast_data1 = tl.broadcast_to(data1, [XBLOCK, RBLOCK])
        broadcast_data2 = tl.broadcast_to(data2, [XBLOCK, RBLOCK])
        
        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_combine(
            running_mean, running_m2, running_weight,
            broadcast_data0, broadcast_data1, broadcast_data2
        )
        
        running_mean = tl.where(r_mask & x_mask, running_mean_next, running_mean)
        running_m2 = tl.where(r_mask & x_mask, running_m2_next, running_m2)
        running_weight = tl.where(r_mask & x_mask, running_weight_next, running_weight)
    
    mean, variance, weight = triton_helpers.welford(running_mean, running_m2, running_weight, 1)
    mean_broadcast = mean[:, None]
    variance_broadcast = variance[:, None]
    
    tl.store(output_ptr0 + (x_indices_flat), mean_broadcast, x_mask)
    
    input3 = tl.load(input_ptr3 + (x_indices_flat), x_mask, eviction_policy='evict_last')
    input4 = tl.load(input_ptr4 + (x_indices_flat), x_mask, eviction_policy='evict_last')
    
    epsilon = 125440.0
    variance_epsilon = 1e-05
    inv_std = tl.extra.cuda.libdevice.rsqrt(variance_broadcast / epsilon + variance_epsilon)
    scale_factor = 1.0000079720023278
    adjusted_mean = mean_broadcast * scale_factor
    momentum = 0.1
    adjusted_mean_scaled = adjusted_mean * momentum
    moving_mean = input3 * 0.9
    updated_mean = adjusted_mean_scaled + moving_mean
    
    moving_variance = input4 * 0.9
    updated_variance = adjusted_mean_scaled + moving_variance
    
    tl.store(output_ptr2 + (x_indices_flat), inv_std, x_mask)
    tl.store(output_ptr4 + (x_indices_flat), updated_mean, x_mask)
    tl.store(output_ptr6 + (x_indices_flat), updated_variance, x_mask)