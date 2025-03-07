# From: 20_MobileNetV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_hardtanh_52poi_fused__native_batch_norm_legit_functional_hardtanh_52(input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, output_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 282240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 576)
    
    input_data = tl.load(input_ptr0 + (x2), xmask)
    mean = tl.load(input_ptr1 + (x0), xmask, eviction_policy='evict_last')
    variance = tl.load(input_ptr2 + (x0), xmask, eviction_policy='evict_last')
    gamma = tl.load(input_ptr3 + (x0), xmask, eviction_policy='evict_last')
    beta = tl.load(input_ptr4 + (x0), xmask, eviction_policy='evict_last')
    
    normalized_data = input_data - mean
    scaled_data = normalized_data * variance
    scaled_and_shifted_data = scaled_data * gamma
    batch_norm_output = scaled_and_shifted_data + beta
    
    clamped_output = triton_helpers.maximum(batch_norm_output, 0.0)
    hardtanh_output = triton_helpers.minimum(clamped_output, 6.0)
    
    tl.store(output_ptr0 + (x2), hardtanh_output, xmask)