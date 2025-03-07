# From: 27_Conv3d_HardSwish_ReLU_Softmax_Mean

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax__softmax_backward_data_div_hardswish_relu_threshold_backward_1poi_fused__softmax__softmax_backward_data_div_hardswish_relu_threshold_backward_1(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, output_ptr0, kernel_size0, kernel_size1, kernel_size2, kernel_size3, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    mod_index = index % kernel_size0
    div_index1 = index // kernel_size1
    div_index0 = index // kernel_size0

    input_value0 = tl.load(input_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    input_value1 = tl.load(
        input_ptr1 + (mod_index + ((-8) * div_index1) + ((-2) * div_index1 * kernel_size3 * kernel_size3) + 4 * kernel_size2 * div_index1 + 8 * kernel_size3 * div_index1 + kernel_size2 * div_index1 * kernel_size3 * kernel_size3 + ((-4) * kernel_size2 * kernel_size3 * div_index1)),
        mask,
        eviction_policy='evict_last'
    )
    input_value2 = tl.load(
        input_ptr2 + (mod_index + ((-8) * div_index1) + ((-2) * div_index1 * kernel_size3 * kernel_size3) + 4 * kernel_size2 * div_index1 + 8 * kernel_size3 * div_index1 + kernel_size2 * div_index1 * kernel_size3 * kernel_size3 + ((-4) * kernel_size2 * kernel_size3 * div_index1)),
        mask,
        eviction_policy='evict_last'
    )
    input_value3 = tl.load(
        input_ptr3 + (mod_index + ((-8) * div_index1) + ((-2) * div_index1 * kernel_size3 * kernel_size3) + 4 * kernel_size2 * div_index1 + 8 * kernel_size3 * div_index1 + kernel_size2 * div_index1 * kernel_size3 * kernel_size3 + ((-4) * kernel_size2 * kernel_size3 * div_index1)),
        mask,
        eviction_policy='evict_last'
    )
    input_value4 = tl.load(input_ptr4 + (div_index0), mask, eviction_policy='evict_last')

    bias = 3.0
    biased_input = input_value0 + bias
    zero = 0.0
    max_value = triton_helpers.maximum(biased_input, zero)
    min_value = 6.0
    clamped_value = triton_helpers.minimum(max_value, min_value)
    hardswish_value = input_value0 * clamped_value
    scale_factor = 0.16666666666666666
    scaled_value = hardswish_value * scale_factor
    zero_tensor = tl.full([1], 0, tl.int32)
    max_scaled_value = triton_helpers.maximum(zero_tensor, scaled_value)
    condition = max_scaled_value <= zero
    diff_value = max_scaled_value - input_value1
    exp_value = tl.math.exp(diff_value)
    softmax_value = exp_value / input_value2
    neg_softmax_value = -softmax_value
    kernel_size0_float = kernel_size0.to(tl.float32)
    scaled_input_value4 = input_value4 / kernel_size0_float
    adjusted_value = scaled_input_value4 * softmax_value
    fused_value = tl.extra.cuda.libdevice.fma(neg_softmax_value, input_value3, adjusted_value)
    final_value = tl.where(condition, zero, fused_value)

    tl.store(output_ptr0 + (linear_index), final_value, mask)