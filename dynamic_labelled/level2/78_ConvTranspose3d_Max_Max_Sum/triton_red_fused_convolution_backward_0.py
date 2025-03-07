# From: 78_ConvTranspose3d_Max_Max_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_0(input_ptr, output_ptr, kernel_size_0, kernel_size_1, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_num_elements = 336
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_x = (input_index % 21)
    input_y = input_index // 21
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_flat_index = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_y = reduction_index

        temp_index_0 = reduction_y + input_x * (triton_helpers.div_floor_integer(
            20 + ((-1) * kernel_size_0) + 2 * kernel_size_0 * kernel_size_0 + 
            ((-8) * kernel_size_1 * kernel_size_0 * kernel_size_0) + 
            ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
            4 * kernel_size_0 * kernel_size_1 + 
            8 * kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1, 
            21
        ))

        temp_index_1 = ((-1) * kernel_size_0) + 2 * kernel_size_0 * kernel_size_0 + 
                       ((-8) * kernel_size_1 * kernel_size_0 * kernel_size_0) + 
                       ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                       4 * kernel_size_0 * kernel_size_1 + 
                       8 * kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1

        temp_mask = temp_index_0 < temp_index_1

        temp_load = tl.load(
            input_ptr + (((-1) * input_y) + 
                         ((-1) * (((reduction_y + input_x * (triton_helpers.div_floor_integer(
                             20 + ((-1) * kernel_size_0) + 2 * kernel_size_0 * kernel_size_0 + 
                             ((-8) * kernel_size_1 * kernel_size_0 * kernel_size_0) + 
                             ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                             4 * kernel_size_0 * kernel_size_1 + 
                             8 * kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1, 
                             21
                         )) // ((-1) + 2 * kernel_size_1)) % ((-1) + 2 * kernel_size_1)))) + 
                         ((-16) * (((reduction_y + input_x * (triton_helpers.div_floor_integer(
                             20 + ((-1) * kernel_size_0) + 2 * kernel_size_0 * kernel_size_0 + 
                             ((-8) * kernel_size_1 * kernel_size_0 * kernel_size_0) + 
                             ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                             4 * kernel_size_0 * kernel_size_1 + 
                             8 * kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1, 
                             21
                         )) // ((-1) + ((-4) * kernel_size_1 * kernel_size_1) + 2 * kernel_size_0 + 
                                  4 * kernel_size_1 + ((-8) * kernel_size_0 * kernel_size_1) + 
                                  8 * kernel_size_0 * kernel_size_1 * kernel_size_1)) % kernel_size_0))) + 
                         ((-64) * kernel_size_1 * kernel_size_1 * 
                          (((reduction_y + input_x * (triton_helpers.div_floor_integer(
                              20 + ((-1) * kernel_size_0) + 2 * kernel_size_0 * kernel_size_0 + 
                              ((-8) * kernel_size_1 * kernel_size_0 * kernel_size_0) + 
                              ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                              4 * kernel_size_0 * kernel_size_1 + 
                              8 * kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1, 
                              21
                          )) // ((-1) + ((-4) * kernel_size_1 * kernel_size_1) + 2 * kernel_size_0 + 
                                   4 * kernel_size_1 + ((-8) * kernel_size_0 * kernel_size_1) + 
                                   8 * kernel_size_0 * kernel_size_1 * kernel_size_1)) % kernel_size_0))) + 
                         ((-4) * kernel_size_1 * 
                          (((reduction_y + input_x * (triton_helpers.div_floor_integer(
                              20 + ((-1) * kernel_size_0) + 2 * kernel_size_0 * kernel_size_0 + 
                              ((-8) * kernel_size_1 * kernel_size_0 * kernel_size_0) + 
                              ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                              4 * kernel_size_0 * kernel_size_1 + 
                              8 * kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1, 
                              21
                          )) // (1 + ((-4) * kernel_size_1) + 4 * kernel_size_1 * kernel_size_1)) % 
                           ((-1) + 2 * kernel_size_0)))) + 
                         ((-4) * input_y * kernel_size_1 * kernel_size_1) + 
                         2 * kernel_size_0 * input_y + 
                         2 * kernel_size_1 * 
                         (((reduction_y + input_x * (triton_helpers.div_floor_integer(
                             20 + ((-1) * kernel_size_0) + 2 * kernel_size_0 * kernel_size_0 + 
                             ((-8) * kernel_size_1 * kernel_size_0 * kernel_size_0) + 
                             ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                             4 * kernel_size_0 * kernel_size_1 + 
                             8 * kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1, 
                             21
                         )) // ((-1) + 2 * kernel_size_1)) % ((-1) + 2 * kernel_size_1))) + 
                         4 * kernel_size_1 * input_y + 
                         4 * kernel_size_1 * kernel_size_1 * 
                         (((reduction_y + input_x * (triton_helpers.div_floor_integer(
                             20 + ((-1) * kernel_size_0) + 2 * kernel_size_0 * kernel_size_0 + 
                             ((-8) * kernel_size_1 * kernel_size_0 * kernel_size_0) + 
                             ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                             4 * kernel_size_0 * kernel_size_1 + 
                             8 * kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1, 
                             21
                         )) // (1 + ((-4) * kernel_size_1) + 4 * kernel_size_1 * kernel_size_1)) % 
                           ((-1) + 2 * kernel_size_0))) + 
                         32 * kernel_size_0 * 
                         (((reduction_y + input_x * (triton_helpers.div_floor_integer(
                             20 + ((-1) * kernel_size_0) + 2 * kernel_size_0 * kernel_size_0 + 
                             ((-8) * kernel_size_1 * kernel_size_0 * kernel_size_0) + 
                             ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                             4 * kernel_size_0 * kernel_size_1 + 
                             8 * kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1, 
                             21
                         )) // ((-1) + ((-4) * kernel_size_1 * kernel_size_1) + 2 * kernel_size_0 + 
                                  4 * kernel_size_1 + ((-8) * kernel_size_0 * kernel_size_1) + 
                                  8 * kernel_size_0 * kernel_size_1 * kernel_size_1)) % kernel_size_0)) + 
                         64 * kernel_size_1 * 
                         (((reduction_y + input_x * (triton_helpers.div_floor_integer(
                             20 + ((-1) * kernel_size_0) + 2 * kernel_size_0 * kernel_size_0 + 
                             ((-8) * kernel_size_1 * kernel_size_0 * kernel_size_0) + 
                             ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                             4 * kernel_size_0 * kernel_size_1 + 
                             8 * kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1, 
                             21
                         )) // ((-1) + ((-4) * kernel_size_1 * kernel_size_1) + 2 * kernel_size_0 + 
                                  4 * kernel_size_1 + ((-8) * kernel_size_0 * kernel_size_1) + 
                                  8 * kernel_size_0 * kernel_size_1 * kernel_size_1)) % kernel_size_0)) + 
                         ((-128) * kernel_size_0 * kernel_size_1 * 
                          (((reduction_y + input_x * (triton_helpers.div_floor_integer(
                              20 + ((-1) * kernel_size_0) + 2 * kernel_size_0 * kernel_size_0 + 
                              ((-8) * kernel_size_1 * kernel_size_0 * kernel_size_0) + 
                              ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                              4 * kernel_size_0 * kernel_size_1 + 
                              8 * kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1, 
                              21
                          )) // ((-1) + ((-4) * kernel_size_1 * kernel_size_1) + 2 * kernel_size_0 + 
                                   4 * kernel_size_1 + ((-8) * kernel_size_0 * kernel_size_1) + 
                                   8 * kernel_size_0 * kernel_size_1 * kernel_size_1)) % kernel_size_0))) + 
                         ((-8) * kernel_size_0 * kernel_size_1 * input_y) + 
                         8 * kernel_size_0 * input_y * kernel_size_1 * kernel_size_1 + 
                         128 * kernel_size_0 * kernel_size_1 * kernel_size_1 * 
                         (((reduction_y + input_x * (triton_helpers.div_floor_integer(
                             20 + ((-1) * kernel_size_0) + 2 * kernel_size_0 * kernel_size_0 + 
                             ((-8) * kernel_size_1 * kernel_size_0 * kernel_size_0) + 
                             ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                             4 * kernel_size_0 * kernel_size_1 + 
                             8 * kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1, 
                             21
                         )) // ((-1) + ((-4) * kernel_size_1 * kernel_size_1) + 2 * kernel_size_0 + 
                                  4 * kernel_size_1 + ((-8) * kernel_size_0 * kernel_size_1) + 
                                  8 * kernel_size_0 * kernel_size_1 * kernel_size_1)) % kernel_size_0)) + 
                         (((reduction_y + input_x * (triton_helpers.div_floor_integer(
                             20 + ((-1) * kernel_size_0) + 2 * kernel_size_0 * kernel_size_0 + 
                             ((-8) * kernel_size_1 * kernel_size_0 * kernel_size_0) + 
                             ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                             4 * kernel_size_0 * kernel_size_1 + 
                             8 * kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1, 
                             21
                         ))) % ((-1) + 2 * kernel_size_1))) + 
                         (((reduction_y + input_x * (triton_helpers.div_floor_integer(
                             20 + ((-1) * kernel_size_0) + 2 * kernel_size_0 * kernel_size_0 + 
                             ((-8) * kernel_size_1 * kernel_size_0 * kernel_size_0) + 
                             ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                             4 * kernel_size_0 * kernel_size_1 + 
                             8 * kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1, 
                             21
                         )) // (1 + ((-4) * kernel_size_1) + 4 * kernel_size_1 * kernel_size_1)) % 
                           ((-1) + 2 * kernel_size_0)))), 
            reduction_mask & temp_mask & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        temp_broadcast = tl.broadcast_to(temp_mask, [XBLOCK, RBLOCK])
        temp_accumulate = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(reduction_mask & input_mask, temp_accumulate, temp_accumulator)

    temp_result = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (input_flat_index), temp_result, input_mask)