# From: 5_AlexNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_12poi_fused_max_pool2d_with_indices_12(input_ptr, output_ptr_values, output_ptr_indices, num_elements_y, num_elements_x, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    num_elements_y = 360
    num_elements_x = 256
    
    y_offset = tl.program_id(1) * YBLOCK
    y_indices = y_offset + tl.arange(0, YBLOCK)[None, :]
    y_mask = y_indices < num_elements_y
    
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    
    x_indices_3d = x_indices
    y_mod_6 = (y_indices % 6)
    y_div_6_mod_6 = ((y_indices // 6) % 6)
    y_div_36 = y_indices // 36
    y_mod_36 = (y_indices % 36)
    y_full_index = y_indices
    
    input_offset_0 = 512 * y_mod_6 + 6656 * y_div_6_mod_6 + 43264 * y_div_36
    input_offset_1 = 256 + input_offset_0
    input_offset_2 = 512 + input_offset_0
    input_offset_3 = 3328 + input_offset_0
    input_offset_4 = 3584 + input_offset_0
    input_offset_5 = 3840 + input_offset_0
    input_offset_6 = 6656 + input_offset_0
    input_offset_7 = 6912 + input_offset_0
    input_offset_8 = 7168 + input_offset_0
    
    tmp0 = tl.load(input_ptr + (x_indices_3d + input_offset_0), x_mask & y_mask, eviction_policy='evict_last')
    tmp1 = tl.load(input_ptr + (x_indices_3d + input_offset_1), x_mask & y_mask, eviction_policy='evict_last')
    tmp2 = tl.load(input_ptr + (x_indices_3d + input_offset_2), x_mask & y_mask, eviction_policy='evict_last')
    tmp3 = tl.load(input_ptr + (x_indices_3d + input_offset_3), x_mask & y_mask, eviction_policy='evict_last')
    tmp4 = tl.load(input_ptr + (x_indices_3d + input_offset_4), x_mask & y_mask, eviction_policy='evict_last')
    tmp5 = tl.load(input_ptr + (x_indices_3d + input_offset_5), x_mask & y_mask, eviction_policy='evict_last')
    tmp6 = tl.load(input_ptr + (x_indices_3d + input_offset_6), x_mask & y_mask, eviction_policy='evict_last')
    tmp7 = tl.load(input_ptr + (x_indices_3d + input_offset_7), x_mask & y_mask, eviction_policy='evict_last')
    tmp8 = tl.load(input_ptr + (x_indices_3d + input_offset_8), x_mask & y_mask, eviction_policy='evict_last')
    
    max_val_1 = triton_helpers.maximum(tmp1, tmp0)
    max_val_2 = triton_helpers.maximum(tmp2, max_val_1)
    max_val_3 = triton_helpers.maximum(tmp3, max_val_2)
    max_val_4 = triton_helpers.maximum(tmp4, max_val_3)
    max_val_5 = triton_helpers.maximum(tmp5, max_val_4)
    max_val_6 = triton_helpers.maximum(tmp6, max_val_5)
    max_val_7 = triton_helpers.maximum(tmp7, max_val_6)
    max_val_8 = triton_helpers.maximum(tmp8, max_val_7)
    
    index_1 = tl.full([1, 1], 1, tl.int8)
    index_0 = tl.full([1], 0, tl.int8)
    index_2 = tl.full([1, 1], 2, tl.int8)
    index_3 = tl.full([1, 1], 3, tl.int8)
    index_4 = tl.full([1, 1], 4, tl.int8)
    index_5 = tl.full([1, 1], 5, tl.int8)
    index_6 = tl.full([1, 1], 6, tl.int8)
    index_7 = tl.full([1, 1], 7, tl.int8)
    index_8 = tl.full([1, 1], 8, tl.int8)
    
    max_index = tl.where(tmp1 > tmp0, index_1, index_0)
    max_index = tl.where(tmp2 > max_val_1, index_2, max_index)
    max_index = tl.where(tmp3 > max_val_2, index_3, max_index)
    max_index = tl.where(tmp4 > max_val_3, index_4, max_index)
    max_index = tl.where(tmp5 > max_val_4, index_5, max_index)
    max_index = tl.where(tmp6 > max_val_5, index_6, max_index)
    max_index = tl.where(tmp7 > max_val_6, index_7, max_index)
    max_index = tl.where(tmp8 > max_val_7, index_8, max_index)
    
    tl.store(output_ptr_values + (y_mod_36 + 36 * x_indices_3d + 9216 * y_div_36), max_val_8, x_mask & y_mask)
    tl.store(output_ptr_indices + (x_indices_3d + 256 * y_full_index), max_index, x_mask & y_mask)