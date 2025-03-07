# From: 22_EfficientNetB0

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_hardtanh_31poi_fused__native_batch_norm_legit_functional_hardtanh_31(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_out, output_ptr, xnumel, XBLOCK: tl.constexpr
):
    xnumel = 1128960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 144)
    
    input_mean = tl.load(input_ptr_mean + (x2), xmask)
    input_var = tl.load(input_ptr_var + (x0), xmask, eviction_policy='evict_last')
    input_scale = tl.load(input_ptr_scale + (x0), xmask, eviction_policy='evict_last')
    input_shift = tl.load(input_ptr_shift + (x0), xmask, eviction_policy='evict_last')
    input_out = tl.load(input_ptr_out + (x0), xmask, eviction_policy='evict_last')
    
    normalized_input = input_mean - input_var
    scaled_input = normalized_input * input_var
    scaled_and_shifted_input = scaled_input * input_scale
    final_output = scaled_and_shifted_input + input_shift
    
    clamped_output = triton_helpers.maximum(final_output, 0.0)
    clamped_output = triton_helpers.minimum(clamped_output, 6.0)
    
    tl.store(output_ptr + (x2), clamped_output, xmask)