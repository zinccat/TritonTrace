# From: 38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax__softmax_backward_data_clamp_ge_le_logical_and_mul_scalar_tensor_where_1poi_fused__softmax__softmax_backward_data_clamp_ge_le_logical_and_mul_scalar_tensor_where_1(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, kernel_size0, kernel_size1, kernel_size2, kernel_size3, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    global_index = index
    local_index0 = index % kernel_size0
    local_index2 = index // kernel_size1

    output_value = tl.load(in_out_ptr0 + (global_index), mask, eviction_policy='evict_last')
    input_value0 = tl.load(in_ptr0 + (local_index0 + kernel_size2 * local_index2 * kernel_size3 * kernel_size3), mask, eviction_policy='evict_last')
    input_value1 = tl.load(in_ptr1 + (local_index0 + kernel_size2 * local_index2 * kernel_size3 * kernel_size3), mask, eviction_policy='evict_last')
    input_value2 = tl.load(in_ptr2 + (local_index0 + kernel_size2 * local_index2 * kernel_size3 * kernel_size3), mask, eviction_policy='evict_last')
    input_value3 = tl.load(in_out_ptr0 + (global_index), mask, eviction_policy='evict_last')

    lower_bound = 0.0
    upper_bound = 1.0

    is_ge_lower = output_value >= lower_bound
    is_le_upper = output_value <= upper_bound
    is_within_bounds = is_ge_lower & is_le_upper

    clamped_value = triton_helpers.maximum(output_value, lower_bound)
    clamped_value = triton_helpers.minimum(clamped_value, upper_bound)

    adjusted_value = clamped_value - input_value0
    exp_value = tl.math.exp(adjusted_value)
    normalized_value = exp_value / input_value1
    neg_normalized_value = -normalized_value

    scale_factor = 2.0
    scaled_input_value3 = input_value3 * scale_factor
    scaled_normalized_value = scaled_input_value3 * normalized_value

    fused_value = tl.extra.cuda.libdevice.fma(neg_normalized_value, input_value2, scaled_normalized_value)

    final_value = tl.where(is_within_bounds, fused_value, lower_bound)

    tl.store(in_out_ptr0 + (global_index), final_value, mask)