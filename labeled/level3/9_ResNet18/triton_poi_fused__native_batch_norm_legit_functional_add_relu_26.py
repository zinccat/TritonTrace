# From: 9_ResNet18

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_relu_26poi_fused__native_batch_norm_legit_functional_add_relu_26(
    in_out_ptr, input_ptr, mean_ptr, inv_std_ptr, weight_ptr, bias_ptr, 
    input_ptr2, mean_ptr2, inv_std_ptr2, weight_ptr2, bias_ptr2, 
    xnumel, XBLOCK: tl.constexpr
):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
    
    input_val = tl.load(input_ptr + (x2), xmask)
    mean_val = tl.load(mean_ptr + (x0), xmask, eviction_policy='evict_last')
    inv_std_val = tl.load(inv_std_ptr + (x0), xmask, eviction_policy='evict_last')
    weight_val = tl.load(weight_ptr + (x0), xmask, eviction_policy='evict_last')
    bias_val = tl.load(bias_ptr + (x0), xmask, eviction_policy='evict_last')
    
    input_val2 = tl.load(input_ptr2 + (x2), xmask)
    mean_val2 = tl.load(mean_ptr2 + (x0), xmask, eviction_policy='evict_last')
    inv_std_val2 = tl.load(inv_std_ptr2 + (x0), xmask, eviction_policy='evict_last')
    weight_val2 = tl.load(weight_ptr2 + (x0), xmask, eviction_policy='evict_last')
    bias_val2 = tl.load(bias_ptr2 + (x0), xmask, eviction_policy='evict_last')
    
    normalized_val = input_val - mean_val
    scale_factor = 392.0
    inv_std_adjusted = inv_std_val / scale_factor
    epsilon = 1e-05
    inv_std_adjusted += epsilon
    inv_std_rsqrt = tl.extra.cuda.libdevice.rsqrt(inv_std_adjusted)
    scaled_val = normalized_val * inv_std_rsqrt
    weighted_val = scaled_val * weight_val
    batch_norm_val = weighted_val + bias_val
    
    normalized_val2 = input_val2 - mean_val2
    inv_std_adjusted2 = inv_std_val2 / scale_factor
    inv_std_adjusted2 += epsilon
    inv_std_rsqrt2 = tl.extra.cuda.libdevice.rsqrt(inv_std_adjusted2)
    scaled_val2 = normalized_val2 * inv_std_rsqrt2
    weighted_val2 = scaled_val2 * weight_val2
    batch_norm_val2 = weighted_val2 + bias_val2
    
    fused_val = batch_norm_val + batch_norm_val2
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_val = triton_helpers.maximum(zero_tensor, fused_val)
    
    tl.store(in_out_ptr + (x2), relu_val, xmask)