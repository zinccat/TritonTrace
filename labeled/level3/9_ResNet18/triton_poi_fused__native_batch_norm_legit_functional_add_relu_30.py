# From: 9_ResNet18

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_relu_30poi_fused__native_batch_norm_legit_functional_add_relu_30(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK: tl.constexpr
):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    
    input_data = tl.load(in_ptr0 + (x2), xmask)
    mean = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    variance = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    gamma = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    beta = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    input_data2 = tl.load(in_ptr5 + (x2), xmask)
    mean2 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    variance2 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    gamma2 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    beta2 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    
    normalized_data = input_data - mean
    variance_epsilon = 1e-05
    normalized_data2 = input_data2 - mean2
    
    inv_std_dev = tl.extra.cuda.libdevice.rsqrt(variance / 98.0 + variance_epsilon)
    inv_std_dev2 = tl.extra.cuda.libdevice.rsqrt(variance2 / 98.0 + variance_epsilon)
    
    scaled_data = normalized_data * inv_std_dev * gamma
    scaled_data2 = normalized_data2 * inv_std_dev2 * gamma2
    
    batch_norm_output = scaled_data + beta
    batch_norm_output2 = scaled_data2 + beta2
    
    fused_output = batch_norm_output + batch_norm_output2
    
    relu_output = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(relu_output, fused_output)
    
    tl.store(in_out_ptr0 + (x2), relu_output, xmask)