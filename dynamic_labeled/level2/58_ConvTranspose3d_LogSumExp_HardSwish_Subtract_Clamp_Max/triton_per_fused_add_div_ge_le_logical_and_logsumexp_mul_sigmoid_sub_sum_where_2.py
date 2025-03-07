# From: 58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_div_ge_le_logical_and_logsumexp_mul_sigmoid_sub_sum_where_2(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, kernel_size0, kernel_size1, kernel_size2, 
    input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < input_num_elements
    r_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    
    x3 = x_index
    r2 = r_index
    x4 = (x_index % kernel_size0)
    x5 = x_index // kernel_size0
    
    input_data0 = tl.load(input_ptr0 + (x3), x_mask, eviction_policy='evict_last')
    input_data1 = tl.load(input_ptr1 + (r2), None, eviction_policy='evict_last')
    
    complex_index = (
        x4 + ((-1) * r2) + ((-16) * x5) + ((-64) * x5 * kernel_size2 * kernel_size2) +
        ((-4) * r2 * kernel_size2 * kernel_size2) + 2 * kernel_size1 * r2 +
        4 * kernel_size2 * r2 + 32 * kernel_size1 * x5 + 64 * kernel_size2 * x5 +
        ((-128) * kernel_size1 * kernel_size2 * x5) + ((-8) * kernel_size1 * kernel_size2 * r2) +
        8 * kernel_size1 * r2 * kernel_size2 * kernel_size2 + 128 * kernel_size1 * x5 * kernel_size2 * kernel_size2
    )
    
    input_data2 = tl.load(input_ptr2 + complex_index, x_mask, eviction_policy='evict_last', other=0.0)
    
    constant1 = 3.0
    added_data = input_data0 + constant1
    sigmoid_result = tl.sigmoid(added_data)
    multiplied_data = input_data0 * sigmoid_result
    
    constant2 = 0.16666666666666666
    scaled_data = multiplied_data * constant2
    
    subtracted_data = scaled_data - input_data1
    
    lower_bound = -1.0
    upper_bound = 1.0
    
    is_greater_equal = subtracted_data >= lower_bound
    is_less_equal = subtracted_data <= upper_bound
    within_bounds = is_greater_equal & is_less_equal
    
    default_value = 0.0
    selected_data = tl.where(within_bounds, input_data2, default_value)
    
    broadcasted_data = tl.broadcast_to(selected_data, [XBLOCK, RBLOCK])
    masked_data = tl.where(x_mask, broadcasted_data, 0)
    
    summed_data = tl.sum(masked_data, 1)[:, None]
    tl.store(output_ptr0 + (x3), summed_data, x_mask)