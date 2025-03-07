# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_77poi_fused_clone_77(input_ptr0, input_ptr1, output_ptr0, y_num_elements, x_num_elements, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    y_num_elements = 7680
    x_num_elements = 49
    y_offset = tl.program_id(1) * YBLOCK
    y_indices = y_offset + tl.arange(0, YBLOCK)[None, :]
    y_mask = y_indices < y_num_elements
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < x_num_elements
    x_position = x_indices
    y_block_index = y_indices // 768
    y_within_block = y_indices % 768
    y_row_index = (y_indices // 32) % 24
    y_full_index = y_indices
    temp0 = tl.load(input_ptr0 + (768 + y_within_block + 2304 * x_position + 112896 * y_block_index), x_mask & y_mask, eviction_policy='evict_last')
    temp1 = tl.load(input_ptr1 + (x_position + 49 * y_row_index + 1184 * y_block_index), x_mask & y_mask, eviction_policy='evict_last')
    epsilon = 1e-12
    temp3 = triton_helpers.maximum(temp1, epsilon)
    temp4 = temp0 / temp3
    tl.store(output_ptr0 + (x_position + 49 * y_full_index), temp4, x_mask & y_mask)