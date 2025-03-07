# From: 62_Matmul_GroupNorm_LeakyReLU_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_leaky_relu_native_group_norm_1(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
    
    input_data = tl.load(in_ptr0 + (x2), xmask)
    mean = tl.load(in_ptr1 + (x2 // 32), xmask, eviction_policy='evict_last')
    variance = tl.load(in_ptr2 + (x2 // 32), xmask, eviction_policy='evict_last')
    gamma = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    beta = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    
    normalized_data = input_data - mean
    scaled_data = normalized_data * variance
    scaled_gamma = scaled_data * gamma
    group_norm_output = scaled_gamma + beta
    
    leaky_relu_threshold = 0.0
    leaky_relu_slope = 0.01
    
    positive_mask = group_norm_output > leaky_relu_threshold
    negative_scaled_output = group_norm_output * leaky_relu_slope
    leaky_relu_output = tl.where(positive_mask, group_norm_output, negative_scaled_output)
    
    final_output = leaky_relu_output + leaky_relu_output
    
    tl.store(in_out_ptr0 + (x2), final_output, xmask)