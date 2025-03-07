# From: 97_Matmul_BatchNorm_BiasAdd_Divide_Swish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_0(
    input_ptr_mean, input_ptr_var, input_ptr_bias, 
    output_ptr_normalized, output_ptr_scale, output_ptr_offset, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 512
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x_indices = xindex
    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r_indices = rindex
        input_data = tl.load(input_ptr_mean + (x_indices + (512 * r_indices)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        broadcast_input = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        
        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_reduce(
            broadcast_input, running_mean, running_m2, running_weight, roffset == 0
        )
        
        running_mean = tl.where(rmask & xmask, running_mean_next, running_mean)
        running_m2 = tl.where(rmask & xmask, running_m2_next, running_m2)
        running_weight = tl.where(rmask & xmask, running_weight_next, running_weight)
    
    mean, variance, weight = triton_helpers.welford(running_mean, running_m2, running_weight, 1)
    mean = mean[:, None]
    variance = variance[:, None]
    
    tl.store(output_ptr_normalized + (x_indices), mean, xmask)
    
    input_scale = tl.load(input_ptr_var + (x_indices), xmask, eviction_policy='evict_last')
    input_offset = tl.load(input_ptr_bias + (x_indices), xmask, eviction_policy='evict_last')
    
    epsilon = 128.0
    variance_epsilon = 1e-05
    normalized_variance = variance / epsilon
    adjusted_variance = normalized_variance + variance_epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    
    scale_factor = 1.0078740157480315
    adjusted_mean = normalized_variance * scale_factor
    swish_factor = 0.1
    swish_adjusted_mean = adjusted_mean * swish_factor
    
    momentum = 0.9
    scaled_input_scale = input_scale * momentum
    swish_scaled_input = swish_adjusted_mean + scaled_input_scale
    
    scaled_input_weight = mean * swish_factor
    scaled_input_offset = input_offset * momentum
    final_offset = scaled_input_weight + scaled_input_offset
    
    tl.store(output_ptr_scale + (x_indices), inv_stddev, xmask)
    tl.store(output_ptr_offset + (x_indices), swish_scaled_input, xmask)
    tl.store(output_ptr_normalized + (x_indices), final_offset, xmask)