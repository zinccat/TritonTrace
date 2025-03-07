# From: 20_MobileNetV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_hardtanh_49poi_fused__native_batch_norm_legit_functional_hardtanh_49(input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, output_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1128960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 576)
    
    input_value = tl.load(input_ptr0 + (x2), xmask)
    mean_value = tl.load(input_ptr1 + (x0), xmask, eviction_policy='evict_last')
    variance_value = tl.load(input_ptr2 + (x0), xmask, eviction_policy='evict_last')
    gamma_value = tl.load(input_ptr3 + (x0), xmask, eviction_policy='evict_last')
    beta_value = tl.load(input_ptr4 + (x0), xmask, eviction_policy='evict_last')
    
    normalized_value = input_value - mean_value
    scaled_value = normalized_value * variance_value
    gamma_scaled_value = scaled_value * gamma_value
    batch_norm_output = gamma_scaled_value + beta_value
    
    clamped_min = 0.0
    clamped_value = triton_helpers.maximum(batch_norm_output, clamped_min)
    clamped_max = 6.0
    final_output = triton_helpers.minimum(clamped_value, clamped_max)
    
    tl.store(output_ptr0 + (x2), final_output, xmask)