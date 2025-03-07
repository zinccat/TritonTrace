# From: 12_VGG19

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_18poi_fused_max_pool2d_with_indices_18(
    input_ptr, output_ptr_values, output_ptr_indices, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr
):
    ynumel = 490
    xnumel = 512
    y_offset = tl.program_id(1) * YBLOCK
    y_index = y_offset + tl.arange(0, YBLOCK)[None, :]
    y_mask = y_index < ynumel
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    x2 = x_index
    y0 = (y_index % 7)
    y1 = y_index // 7
    y5 = y_index
    y4 = y_index // 49
    y6 = (y_index % 49)

    input_slice_0 = tl.load(input_ptr + (x2 + 1024 * y0 + 14336 * y1), x_mask & y_mask, eviction_policy='evict_last')
    input_slice_1 = tl.load(input_ptr + (512 + x2 + 1024 * y0 + 14336 * y1), x_mask & y_mask, eviction_policy='evict_last')
    input_slice_7 = tl.load(input_ptr + (7168 + x2 + 1024 * y0 + 14336 * y1), x_mask & y_mask, eviction_policy='evict_last')
    input_slice_12 = tl.load(input_ptr + (7680 + x2 + 1024 * y0 + 14336 * y1), x_mask & y_mask, eviction_policy='evict_last')

    is_greater_1 = input_slice_1 > input_slice_0
    mask_1 = tl.full([1, 1], 1, tl.int8)
    mask_0 = tl.full([1, 1], 0, tl.int8)
    index_mask_1 = tl.where(is_greater_1, mask_1, mask_0)
    max_1 = triton_helpers.maximum(input_slice_1, input_slice_0)

    is_greater_7 = input_slice_7 > max_1
    mask_7 = tl.full([1, 1], 2, tl.int8)
    index_mask_7 = tl.where(is_greater_7, mask_7, index_mask_1)
    max_7 = triton_helpers.maximum(input_slice_7, max_1)

    is_greater_12 = input_slice_12 > max_7
    mask_12 = tl.full([1, 1], 3, tl.int8)
    index_mask_12 = tl.where(is_greater_12, mask_12, index_mask_7)
    max_12 = triton_helpers.maximum(input_slice_12, max_7)

    tl.store(output_ptr_values + (x2 + 512 * y5), index_mask_12, x_mask & y_mask)
    tl.store(output_ptr_indices + (y6 + 49 * x2 + 25088 * y4), max_12, x_mask & y_mask)