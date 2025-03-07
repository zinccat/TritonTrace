# From: 44_Average_Pooling_1D

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_avg_pool2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    valid_mask = tl.full([XBLOCK], True, tl.int1)
    base_index = xindex % 64
    global_index = xindex
    zero_value = tl.full([1], 0, tl.int64)
    is_non_negative = zero_value >= zero_value
    one_value = tl.full([1], 1, tl.int64)
    is_less_than_one = zero_value < one_value
    valid_range_mask = is_non_negative & is_less_than_one
    adjusted_index = (-1) + (2 * base_index)
    is_non_negative_adjusted = adjusted_index >= zero_value
    max_index_value = tl.full([1], 128, tl.int64)
    is_less_than_max = adjusted_index < max_index_value
    valid_adjusted_mask = is_non_negative_adjusted & is_less_than_max
    combined_valid_mask = valid_range_mask & valid_adjusted_mask
    load_value1 = tl.load(in_ptr0 + ((-1) + (2 * global_index)), combined_valid_mask, eviction_policy='evict_last', other=0.0)
    
    double_base_index = 2 * base_index
    is_non_negative_double = double_base_index >= zero_value
    is_less_than_max_double = double_base_index < max_index_value
    valid_double_mask = is_non_negative_double & is_less_than_max_double
    combined_double_valid_mask = valid_range_mask & valid_double_mask
    load_value2 = tl.load(in_ptr0 + (2 * global_index), combined_double_valid_mask, eviction_policy='evict_last', other=0.0)
    
    sum_values1 = load_value2 + load_value1
    
    incremented_base_index = 1 + (2 * base_index)
    is_non_negative_incremented = incremented_base_index >= zero_value
    is_less_than_max_incremented = incremented_base_index < max_index_value
    valid_incremented_mask = is_non_negative_incremented & is_less_than_max_incremented
    combined_incremented_valid_mask = valid_range_mask & valid_incremented_mask
    load_value3 = tl.load(in_ptr0 + (1 + (2 * global_index)), combined_incremented_valid_mask, eviction_policy='evict_last', other=0.0)
    
    sum_values2 = load_value3 + sum_values1
    
    further_incremented_base_index = 2 + (2 * base_index)
    is_non_negative_further = further_incremented_base_index >= zero_value
    is_less_than_max_further = further_incremented_base_index < max_index_value
    valid_further_mask = is_non_negative_further & is_less_than_max_further
    combined_further_valid_mask = valid_range_mask & valid_further_mask
    load_value4 = tl.load(in_ptr0 + (2 + (2 * global_index)), combined_further_valid_mask, eviction_policy='evict_last', other=0.0)
    
    sum_values3 = load_value4 + sum_values2
    
    divisor = 1 + ((-2) * base_index) + ((129) * ((129) <= (3 + (2 * base_index))) + (3 + (2 * base_index)) * ((3 + (2 * base_index)) < (129)))
    average_value = sum_values3 / divisor
    
    tl.store(out_ptr0 + (global_index), average_value, None)