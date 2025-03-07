# From: 44_Average_Pooling_1D

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool2d_0poi_fused_avg_pool2d_0(in_ptr0, out_ptr0, kernel_size_x, kernel_size_y, total_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < total_elements
    x_mod_kernel = index % kernel_size_x
    x_div_kernel = index // kernel_size_x
    flat_index = index
    zero_mask = tl.full([1], 0, tl.int64)
    valid_mask = zero_mask >= zero_mask
    one_mask = tl.full([1], 1, tl.int64)
    within_bounds_mask = zero_mask < one_mask
    base_mask = valid_mask & within_bounds_mask
    adjusted_index = (-1) + 2 * x_mod_kernel
    valid_adjusted_index = adjusted_index >= zero_mask
    within_kernel_y = adjusted_index < kernel_size_y
    valid_adjusted_index_mask = valid_adjusted_index & within_kernel_y
    combined_mask = base_mask & valid_adjusted_index_mask
    value1 = tl.load(in_ptr0 + ((-1) + 2 * x_mod_kernel + kernel_size_y * x_div_kernel), combined_mask & mask, eviction_policy='evict_last', other=0.0)
    
    adjusted_index2 = 2 * x_mod_kernel
    valid_adjusted_index2 = adjusted_index2 >= zero_mask
    within_kernel_y2 = adjusted_index2 < kernel_size_y
    valid_adjusted_index_mask2 = valid_adjusted_index2 & within_kernel_y2
    combined_mask2 = base_mask & valid_adjusted_index_mask2
    value2 = tl.load(in_ptr0 + (2 * x_mod_kernel + kernel_size_y * x_div_kernel), combined_mask2 & mask, eviction_policy='evict_last', other=0.0)
    
    sum_values = value2 + value1
    
    adjusted_index3 = 1 + 2 * x_mod_kernel
    valid_adjusted_index3 = adjusted_index3 >= zero_mask
    within_kernel_y3 = adjusted_index3 < kernel_size_y
    valid_adjusted_index_mask3 = valid_adjusted_index3 & within_kernel_y3
    combined_mask3 = base_mask & valid_adjusted_index_mask3
    value3 = tl.load(in_ptr0 + (1 + 2 * x_mod_kernel + kernel_size_y * x_div_kernel), combined_mask3 & mask, eviction_policy='evict_last', other=0.0)
    
    sum_values += value3
    
    adjusted_index4 = 2 + 2 * x_mod_kernel
    valid_adjusted_index4 = adjusted_index4 >= zero_mask
    within_kernel_y4 = adjusted_index4 < kernel_size_y
    valid_adjusted_index_mask4 = valid_adjusted_index4 & within_kernel_y4
    combined_mask4 = base_mask & valid_adjusted_index_mask4
    value4 = tl.load(in_ptr0 + (2 + 2 * x_mod_kernel + kernel_size_y * x_div_kernel), combined_mask4 & mask, eviction_policy='evict_last', other=0.0)
    
    sum_values += value4
    
    divisor = 1 + ((-2) * x_mod_kernel) + ((1 + kernel_size_y) * ((1 + kernel_size_y) <= (3 + 2 * x_mod_kernel)) + (3 + 2 * x_mod_kernel) * ((3 + 2 * x_mod_kernel) < (1 + kernel_size_y)))
    average_value = sum_values / divisor
    
    tl.store(out_ptr0 + (flat_index), average_value, mask)