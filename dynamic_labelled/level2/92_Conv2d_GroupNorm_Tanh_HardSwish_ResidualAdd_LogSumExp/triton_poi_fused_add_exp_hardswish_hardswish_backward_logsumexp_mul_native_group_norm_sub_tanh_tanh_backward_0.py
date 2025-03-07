# From: 92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_exp_hardswish_hardswish_backward_logsumexp_mul_native_group_norm_sub_tanh_tanh_backward_0poi_fused_add_exp_hardswish_hardswish_backward_logsumexp_mul_native_group_norm_sub_tanh_tanh_backward_0(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, 
    output_ptr0, output_ptr1, kernel_size0, kernel_size1, kernel_size2, kernel_size3, 
    num_elements, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < num_elements

    x3 = x_index
    x5 = x_index // kernel_size0
    x1 = ((x_index // kernel_size1) % 16)
    x4 = (x_index % kernel_size0)
    x7 = x_index // kernel_size2

    input_val0 = tl.load(input_ptr0 + (x3), x_mask, eviction_policy='evict_last')
    input_val1 = tl.load(input_ptr1 + (x5 // 2), x_mask, eviction_policy='evict_last')
    input_val2 = tl.load(input_ptr2 + (x5 // 2), x_mask, eviction_policy='evict_last')
    input_val3 = tl.load(input_ptr3 + (x1), x_mask, eviction_policy='evict_last')
    input_val4 = tl.load(input_ptr4 + (x1), x_mask, eviction_policy='evict_last')
    input_val5 = tl.load(input_ptr5 + (x4 + 4*x7 + x7*kernel_size3*kernel_size3 + ((-4)*kernel_size3*x7)), x_mask, eviction_policy='evict_last')
    input_val6 = tl.load(input_ptr6 + (x4 + 4*x7 + x7*kernel_size3*kernel_size3 + ((-4)*kernel_size3*x7)), x_mask, eviction_policy='evict_last')

    intermediate_val1 = input_val0 - input_val1
    intermediate_val2 = intermediate_val1 * input_val2
    intermediate_val3 = intermediate_val2 * input_val3
    intermediate_val4 = intermediate_val3 + input_val4
    tanh_result = tl.extra.cuda.libdevice.tanh(intermediate_val4)

    tanh_less_than_neg3 = tanh_result < -3.0
    tanh_less_equal_3 = tanh_result <= 3.0
    tanh_plus_3 = tanh_result + 3.0
    zero = 0.0
    max_tanh_plus_3 = triton_helpers.maximum(tanh_plus_3, zero)
    min_tanh_plus_3 = triton_helpers.minimum(max_tanh_plus_3, 6.0)
    scaled_tanh = tanh_result * min_tanh_plus_3
    scaled_tanh_times_1_6 = scaled_tanh * 0.16666666666666666
    adjusted_input = input_val0 + scaled_tanh_times_1_6
    exp_input = adjusted_input - input_val6
    exp_result = tl.math.exp(exp_input)
    intermediate_val5 = input_val5 * exp_result
    scaled_tanh_times_0_166 = tanh_result * 0.16666666666666666
    intermediate_val6 = scaled_tanh_times_0_166 + 0.5
    final_result1 = intermediate_val5 * intermediate_val6
    final_result2 = tl.where(tanh_less_equal_3, final_result1, intermediate_val5)
    final_result3 = tl.where(tanh_less_than_neg3, zero, final_result2)

    tanh_squared = tanh_result * tanh_result
    one_minus_tanh_squared = 1.0 - tanh_squared
    final_output = final_result3 * one_minus_tanh_squared

    tl.store(output_ptr0 + (x3), tanh_result, x_mask)
    tl.store(output_ptr1 + (x3), final_output, x_mask)