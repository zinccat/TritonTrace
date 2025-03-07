# From: 11_VGG16

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_18poi_fused_max_pool2d_with_indices_18(
    input_ptr, output_ptr_values, output_ptr_indices, 
    y_num_elements, x_num_elements, 
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr
):
    y_num_elements = 490
    x_num_elements = 512

    y_offset = tl.program_id(1) * YBLOCK
    y_indices = y_offset + tl.arange(0, YBLOCK)[None, :]
    y_mask = y_indices < y_num_elements

    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < x_num_elements

    x_indices_flat = x_indices
    y_mod_7 = y_indices % 7
    y_div_7 = y_indices // 7
    y_flat = y_indices
    y_div_49 = y_indices // 49
    y_mod_49 = y_indices % 49

    input_slice_0 = tl.load(input_ptr + (x_indices_flat + 1024 * y_mod_7 + 14336 * y_div_7), x_mask & y_mask, eviction_policy='evict_last')
    input_slice_1 = tl.load(input_ptr + (512 + x_indices_flat + 1024 * y_mod_7 + 14336 * y_div_7), x_mask & y_mask, eviction_policy='evict_last')
    input_slice_7 = tl.load(input_ptr + (7168 + x_indices_flat + 1024 * y_mod_7 + 14336 * y_div_7), x_mask & y_mask, eviction_policy='evict_last')
    input_slice_12 = tl.load(input_ptr + (7680 + x_indices_flat + 1024 * y_mod_7 + 14336 * y_div_7), x_mask & y_mask, eviction_policy='evict_last')

    is_greater_1 = input_slice_1 > input_slice_0
    mask_1 = tl.full([1, 1], 1, tl.int8)
    mask_0 = tl.full([1, 1], 0, tl.int8)
    max_index_1 = tl.where(is_greater_1, mask_1, mask_0)
    max_value_1 = triton_helpers.maximum(input_slice_1, input_slice_0)

    is_greater_7 = input_slice_7 > max_value_1
    mask_7 = tl.full([1, 1], 2, tl.int8)
    max_index_7 = tl.where(is_greater_7, mask_7, max_index_1)
    max_value_7 = triton_helpers.maximum(input_slice_7, max_value_1)

    is_greater_12 = input_slice_12 > max_value_7
    mask_12 = tl.full([1, 1], 3, tl.int8)
    max_index_12 = tl.where(is_greater_12, mask_12, max_index_7)
    max_value_12 = triton_helpers.maximum(input_slice_12, max_value_7)

    tl.store(output_ptr_values + (x_indices_flat + 512 * y_flat), max_index_12, x_mask & y_mask)
    tl.store(output_ptr_indices + (y_mod_49 + 49 * x_indices_flat + 25088 * y_div_49), max_value_12, x_mask & y_mask)