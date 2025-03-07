# From: 23_EfficientNetB1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_57poi_fused__native_batch_norm_legit_functional_57(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_x, 
    output_ptr, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    element_index = index
    channel_index = index % 192

    mean = tl.load(input_ptr_mean + (element_index), None)
    variance = tl.load(input_ptr_var + (channel_index), None, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (channel_index), None, eviction_policy='evict_last')
    bias = tl.load(input_ptr_bias + (channel_index), None, eviction_policy='evict_last')
    x = tl.load(input_ptr_x + (element_index), None)

    centered_x = x - mean
    variance_epsilon = 1e-05
    normalized_variance = variance / 640.0 + variance_epsilon
    inv_std_dev = tl.extra.cuda.libdevice.rsqrt(normalized_variance)

    normalized_x = centered_x * inv_std_dev
    scaled_x = normalized_x * scale
    output = scaled_x + bias

    tl.store(output_ptr + (element_index), output, None)