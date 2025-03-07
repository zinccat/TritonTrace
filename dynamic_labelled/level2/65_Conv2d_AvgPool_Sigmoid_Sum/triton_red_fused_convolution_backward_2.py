# From: 65_Conv2d_AvgPool_Sigmoid_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_2(input_ptr, output_ptr, kernel_size_0, kernel_size_1, kernel_size_2, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_num_elements = 240
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_1 = input_index // 16
    input_index_0 = (input_index % 16)
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_index_3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_2 = reduction_index

        temp_index_0 = reduction_index_2 + input_index_1 * (triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15))
        temp_index_1 = 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1
        temp_mask = temp_index_0 < temp_index_1

        load_index = (
            -2 * (((reduction_index_2 + input_index_1 * (triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)) // kernel_size_2) % kernel_size_2)) +
            4 * input_index_0 +
            64 * (((reduction_index_2 + input_index_1 * (triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)) // (4 + kernel_size_1 * kernel_size_1 + (-4) * kernel_size_1)) % kernel_size_0)) +
            kernel_size_1 * (((reduction_index_2 + input_index_1 * (triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)) // kernel_size_2) % kernel_size_2)) +
            input_index_0 * kernel_size_1 * kernel_size_1 +
            (-64) * kernel_size_1 * (((reduction_index_2 + input_index_1 * (triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)) // (4 + kernel_size_1 * kernel_size_1 + (-4) * kernel_size_1)) % kernel_size_0)) +
            (-4) * kernel_size_1 * input_index_0 +
            16 * kernel_size_1 * kernel_size_1 * (((reduction_index_2 + input_index_1 * (triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)) // (4 + kernel_size_1 * kernel_size_1 + (-4) * kernel_size_1)) % kernel_size_0)) +
            (reduction_index_2 + input_index_1 * (triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15))) % kernel_size_2
        )

        loaded_values = tl.load(input_ptr + load_index, reduction_mask & temp_mask & input_mask, eviction_policy='evict_last', other=0.0)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        temp_accumulator += broadcasted_values
        temp_accumulator = tl.where(reduction_mask & input_mask, temp_accumulator, temp_accumulator)

    summed_values = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (input_index_3), summed_values, input_mask)