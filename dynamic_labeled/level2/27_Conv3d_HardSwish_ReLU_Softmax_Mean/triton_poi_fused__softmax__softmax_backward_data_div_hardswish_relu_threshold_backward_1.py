# From: 27_Conv3d_HardSwish_ReLU_Softmax_Mean

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax__softmax_backward_data_div_hardswish_relu_threshold_backward_1(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, output_ptr0, kernel_size0, kernel_size1, kernel_size2, kernel_size3, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    mod_index = index % kernel_size0
    div_index1 = index // kernel_size1
    div_index2 = index // kernel_size0

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
    input_value4 = tl.load(input_ptr4 + (div_index2), mask, eviction_policy='evict_last')

    constant1 = 3.0
    adjusted_input = input_value0 + constant1
    zero = 0.0
    max_value = triton_helpers.maximum(adjusted_input, zero)
    constant2 = 6.0
    clamped_value = triton_helpers.minimum(max_value, constant2)
    hardswish_value = input_value0 * clamped_value
    constant3 = 0.16666666666666666
    scaled_value = hardswish_value * constant3
    zero_tensor = tl.full([1], 0, tl.int32)
    max_scaled_value = triton_helpers.maximum(zero_tensor, scaled_value)
    condition = max_scaled_value <= zero
    difference = max_scaled_value - input_value1
    exp_value = tl.math.exp(difference)
    division_result = exp_value / input_value2
    neg_division_result = -division_result
    constant4 = kernel_size0
    float_constant4 = constant4.to(tl.float32)
    scaled_input_value4 = input_value4 / float_constant4
    scaled_exp_value = scaled_input_value4 * exp_value
    fused_multiply_add = tl.extra.cuda.libdevice.fma(neg_division_result, input_value3, scaled_exp_value)
    final_value = tl.where(condition, zero, fused_multiply_add)

    tl.store(output_ptr0 + (linear_index), final_value, mask)