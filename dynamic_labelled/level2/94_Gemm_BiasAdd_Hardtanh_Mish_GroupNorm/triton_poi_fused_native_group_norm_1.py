# From: 94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_1poi_fused_native_group_norm_1(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, 
    output_ptr0, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    index2 = index
    index0 = index % 1024

    input_val0 = tl.load(input_ptr0 + (index2), mask)
    input_val1 = tl.load(input_ptr1 + (index0), mask, eviction_policy='evict_last')
    input_val2 = tl.load(input_ptr2 + (index2 // 32), mask, eviction_policy='evict_last')
    input_val3 = tl.load(input_ptr3 + (index2 // 32), mask, eviction_policy='evict_last')
    input_val4 = tl.load(input_ptr4 + (index0), mask, eviction_policy='evict_last')
    input_val5 = tl.load(input_ptr5 + (index0), mask, eviction_policy='evict_last')

    sum_input = input_val0 + input_val1
    min_value = -1.0
    max_clamped = triton_helpers.maximum(sum_input, min_value)
    max_value = 1.0
    clamped_value = triton_helpers.minimum(max_clamped, max_value)
    threshold = 20.0
    is_greater_than_threshold = clamped_value > threshold
    exp_value = tl.math.exp(clamped_value)
    log1p_value = tl.extra.cuda.libdevice.log1p(exp_value)
    tanh_input = tl.where(is_greater_than_threshold, clamped_value, log1p_value)
    tanh_value = tl.extra.cuda.libdevice.tanh(tanh_input)
    mish_output = clamped_value * tanh_value

    normalized_value = mish_output - input_val2
    scale_factor = 32.0
    scaled_value = input_val3 / scale_factor
    epsilon = 1e-05
    adjusted_value = scaled_value + epsilon
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(adjusted_value)
    scaled_normalized = normalized_value * reciprocal_sqrt
    final_output = scaled_normalized * input_val4 + input_val5

    tl.store(output_ptr0 + (index2), final_output, mask)