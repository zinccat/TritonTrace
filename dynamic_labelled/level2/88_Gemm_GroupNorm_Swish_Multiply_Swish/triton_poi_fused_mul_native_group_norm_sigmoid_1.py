# From: 88_Gemm_GroupNorm_Swish_Multiply_Swish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mul_native_group_norm_sigmoid_1(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1024)
    
    input_data = tl.load(in_ptr0 + (x2), xmask)
    mean = tl.load(in_ptr1 + (x2 // 64), xmask, eviction_policy='evict_last')
    variance = tl.load(in_ptr2 + (x2 // 64), xmask, eviction_policy='evict_last')
    gamma = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    beta = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    swish_coeff = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    
    normalized_data = input_data - mean
    scaled_data = normalized_data * variance
    gamma_scaled_data = scaled_data * gamma
    beta_adjusted_data = gamma_scaled_data + beta
    
    sigmoid_output = tl.sigmoid(beta_adjusted_data)
    swish_input = beta_adjusted_data * sigmoid_output
    swish_output = swish_input * swish_coeff
    
    tl.store(in_out_ptr0 + (x2), swish_output, xmask)