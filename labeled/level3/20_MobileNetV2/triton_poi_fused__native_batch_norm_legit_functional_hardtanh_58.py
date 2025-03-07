# From: 20_MobileNetV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_hardtanh_58poi_fused__native_batch_norm_legit_functional_hardtanh_58(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, output_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xnumel = 470400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 960)
    
    input_value0 = tl.load(input_ptr0 + (x2), xmask)
    mean = tl.load(input_ptr1 + (x0), xmask, eviction_policy='evict_last')
    variance = tl.load(input_ptr2 + (x0), xmask, eviction_policy='evict_last')
    gamma = tl.load(input_ptr3 + (x0), xmask, eviction_policy='evict_last')
    beta = tl.load(input_ptr4 + (x0), xmask, eviction_policy='evict_last')
    
    normalized_value = input_value0 - mean
    scaled_value = normalized_value * variance
    scaled_gamma = scaled_value * gamma
    batch_norm_result = scaled_gamma + beta
    
    clamped_min = 0.0
    clamped_value = triton_helpers.maximum(batch_norm_result, clamped_min)
    clamped_max = 6.0
    final_result = triton_helpers.minimum(clamped_value, clamped_max)
    
    tl.store(output_ptr0 + (x2), final_result, xmask)