# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_3poi_fused_clone_3(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, 
    output_ptr0, y_num_elements, x_num_elements, 
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr
):
    y_num_elements = 94080
    x_num_elements = 32

    y_offset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    y_index = y_offset + tl.arange(0, YBLOCK)[None, :]
    y_mask = y_index < y_num_elements

    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements

    x_coord = x_index
    y_coord_0 = (y_index % 49)
    y_coord_1 = ((y_index // 49) % 3)
    y_coord_2 = y_index // 147
    y_coord_4 = y_index

    tmp0 = tl.load(
        input_ptr0 + (
            7 * ((y_coord_2 % 8)) + 
            56 * (y_coord_0 // 7) + 
            392 * (((y_coord_2 // 8) % 8)) + 
            3136 * x_coord + 
            100352 * y_coord_1 + 
            301056 * (y_coord_2 // 64) + 
            ((y_coord_0 % 7))
        ), 
        x_mask & y_mask, 
        eviction_policy='evict_last'
    )

    tmp1 = tl.load(
        input_ptr1 + (
            7 * ((y_coord_2 % 8)) + 
            56 * (y_coord_0 // 7) + 
            392 * (y_coord_2 // 8) + 
            ((y_coord_0 % 7))
        ), 
        y_mask, 
        eviction_policy='evict_last'
    )

    tmp3 = tl.load(
        input_ptr2 + (
            7 * ((y_coord_2 % 8)) + 
            56 * (y_coord_0 // 7) + 
            392 * (y_coord_2 // 8) + 
            ((y_coord_0 % 7))
        ), 
        y_mask, 
        eviction_policy='evict_last'
    )

    tmp5 = tl.load(
        input_ptr3 + (x_coord + 32 * y_coord_1), 
        x_mask & y_mask, 
        eviction_policy='evict_last'
    )

    tmp7 = tl.load(
        input_ptr4 + (x_coord + 32 * y_coord_1), 
        x_mask & y_mask, 
        eviction_policy='evict_last'
    )

    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7

    tl.store(
        output_ptr0 + (x_coord + 32 * y_coord_4), 
        tmp8, 
        x_mask & y_mask
    )