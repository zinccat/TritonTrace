# From: 44_Average_Pooling_1D

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool2d_0(in_ptr0, out_ptr0, kernel_size_x, kernel_size_y, total_elements, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_indices < total_elements
    x_mod_kernel = x_indices % kernel_size_x
    x_div_kernel = x_indices // kernel_size_x
    x_full_index = x_indices

    # Temporary variables for logical operations
    is_valid_index = tl.full([1], 0, tl.int64)
    is_non_negative = is_valid_index >= is_valid_index
    is_less_than_one = is_valid_index < tl.full([1], 1, tl.int64)
    valid_index_mask = is_non_negative & is_less_than_one

    # Load and accumulate values
    load_index_1 = (-1) + 2 * x_mod_kernel
    is_within_bounds_1 = (load_index_1 >= is_valid_index) & (load_index_1 < kernel_size_y)
    valid_load_mask_1 = valid_index_mask & is_within_bounds_1
    value_1 = tl.load(in_ptr0 + (load_index_1 + kernel_size_y * x_div_kernel), valid_load_mask_1 & x_mask, eviction_policy='evict_last', other=0.0)

    load_index_2 = 2 * x_mod_kernel
    is_within_bounds_2 = (load_index_2 >= is_valid_index) & (load_index_2 < kernel_size_y)
    valid_load_mask_2 = valid_index_mask & is_within_bounds_2
    value_2 = tl.load(in_ptr0 + (load_index_2 + kernel_size_y * x_div_kernel), valid_load_mask_2 & x_mask, eviction_policy='evict_last', other=0.0)

    load_index_3 = 1 + 2 * x_mod_kernel
    is_within_bounds_3 = (load_index_3 >= is_valid_index) & (load_index_3 < kernel_size_y)
    valid_load_mask_3 = valid_index_mask & is_within_bounds_3
    value_3 = tl.load(in_ptr0 + (load_index_3 + kernel_size_y * x_div_kernel), valid_load_mask_3 & x_mask, eviction_policy='evict_last', other=0.0)

    load_index_4 = 2 + 2 * x_mod_kernel
    is_within_bounds_4 = (load_index_4 >= is_valid_index) & (load_index_4 < kernel_size_y)
    valid_load_mask_4 = valid_index_mask & is_within_bounds_4
    value_4 = tl.load(in_ptr0 + (load_index_4 + kernel_size_y * x_div_kernel), valid_load_mask_4 & x_mask, eviction_policy='evict_last', other=0.0)

    # Accumulate values
    accumulated_value = value_2 + value_1
    accumulated_value += value_3
    accumulated_value += value_4

    # Calculate divisor for average
    divisor = 1 + ((-2) * x_mod_kernel) + ((1 + kernel_size_y) * ((1 + kernel_size_y) <= (3 + 2 * x_mod_kernel)) + (3 + 2 * x_mod_kernel) * ((3 + 2 * x_mod_kernel) < (1 + kernel_size_y)))

    # Compute average and store result
    average_value = accumulated_value / divisor
    tl.store(out_ptr0 + (x_full_index), average_value, x_mask)