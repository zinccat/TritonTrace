# From: 92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_exp_hardswish_hardswish_backward_logsumexp_mul_native_group_norm_sub_tanh_tanh_backward_0(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, 
    output_ptr0, output_ptr1, kernel_size0, kernel_size1, kernel_size2, kernel_size3, 
    num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    group_index = index // kernel_size0
    channel_index = ((index // kernel_size1) % 16)
    spatial_index = (index % kernel_size0)
    batch_index = index // kernel_size2

    input_val0 = tl.load(input_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    input_val1 = tl.load(input_ptr1 + (group_index // 2), mask, eviction_policy='evict_last')
    input_val2 = tl.load(input_ptr2 + (group_index // 2), mask, eviction_policy='evict_last')
    input_val3 = tl.load(input_ptr3 + (channel_index), mask, eviction_policy='evict_last')
    input_val4 = tl.load(input_ptr4 + (channel_index), mask, eviction_policy='evict_last')
    input_val5 = tl.load(input_ptr5 + (spatial_index + 4 * batch_index + batch_index * kernel_size3 * kernel_size3 + ((-4) * kernel_size3 * batch_index)), mask, eviction_policy='evict_last')
    input_val6 = tl.load(input_ptr6 + (spatial_index + 4 * batch_index + batch_index * kernel_size3 * kernel_size3 + ((-4) * kernel_size3 * batch_index)), mask, eviction_policy='evict_last')

    intermediate_val1 = input_val0 - input_val1
    intermediate_val2 = intermediate_val1 * input_val2
    intermediate_val3 = intermediate_val2 * input_val3
    intermediate_val4 = intermediate_val3 + input_val4
    tanh_val = tl.extra.cuda.libdevice.tanh(intermediate_val4)

    tanh_less_than_neg3 = tanh_val < -3.0
    tanh_less_equal_3 = tanh_val <= 3.0
    tanh_plus_3 = tanh_val + 3.0
    zero_val = 0.0
    max_val = triton_helpers.maximum(tanh_plus_3, zero_val)
    min_val = triton_helpers.minimum(max_val, 6.0)
    scaled_tanh = tanh_val * min_val
    scaled_tanh_times_1_6 = scaled_tanh * 0.16666666666666666
    adjusted_input = input_val0 + scaled_tanh_times_1_6
    exp_input = adjusted_input - input_val6
    exp_val = tl.math.exp(exp_input)
    intermediate_val5 = input_val5 * exp_val
    scaled_tanh_times_0_166 = tanh_val * 0.16666666666666666
    intermediate_val6 = scaled_tanh_times_0_166 + 0.5
    conditional_val = tl.where(tanh_less_equal_3, intermediate_val5 * intermediate_val6, intermediate_val5)
    final_val = tl.where(tanh_less_than_neg3, zero_val, conditional_val)
    squared_tanh = tanh_val * tanh_val
    one_val = 1.0
    one_minus_squared_tanh = one_val - squared_tanh
    output_val = final_val * one_minus_squared_tanh

    tl.store(output_ptr0 + (linear_index), tanh_val, mask)
    tl.store(output_ptr1 + (linear_index), output_val, mask)