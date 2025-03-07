# From: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_5(
    input_ptr, output_ptr, kernel_size_0, kernel_size_1, kernel_size_2, kernel_size_3, 
    input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 3984
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_0 = (input_index % 249)
    input_1 = input_index // 249
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_2 = reduction_index

        temp_index_0 = reduction_2 + input_0 * (
            triton_helpers.div_floor_integer(
                248 + ((-1) * kernel_size_0) + ((-12) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                6 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1 * kernel_size_1, 
                249
            )
        )

        temp_index_1 = ((-1) * kernel_size_0) + ((-12) * kernel_size_0 * kernel_size_1 * kernel_size_1) + \
                       6 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1 * kernel_size_1

        temp_mask = temp_index_0 < temp_index_1

        temp_load = tl.load(
            input_ptr + (
                ((-1) * input_1) + 
                ((-1) * (((temp_index_0 // kernel_size_2) % kernel_size_2))) + 
                ((-16) * (((temp_index_0 // ((-1) + ((-12) * kernel_size_1 * kernel_size_1) + 6 * kernel_size_1 + 8 * kernel_size_1 * kernel_size_1 * kernel_size_1)) % kernel_size_0))) + 
                ((-192) * kernel_size_1 * kernel_size_1 * (((temp_index_0 // ((-1) + ((-12) * kernel_size_1 * kernel_size_1) + 6 * kernel_size_1 + 8 * kernel_size_1 * kernel_size_1 * kernel_size_1)) % kernel_size_0))) + 
                ((-12) * input_1 * kernel_size_1 * kernel_size_1) + 
                ((-4) * kernel_size_1 * (((temp_index_0 // kernel_size_3) % kernel_size_2))) + 
                2 * kernel_size_1 * (((temp_index_0 // kernel_size_2) % kernel_size_2)) + 
                4 * kernel_size_1 * kernel_size_1 * (((temp_index_0 // kernel_size_3) % kernel_size_2)) + 
                6 * kernel_size_1 * input_1 + 
                8 * input_1 * kernel_size_1 * kernel_size_1 * kernel_size_1 + 
                96 * kernel_size_1 * (((temp_index_0 // ((-1) + ((-12) * kernel_size_1 * kernel_size_1) + 6 * kernel_size_1 + 8 * kernel_size_1 * kernel_size_1 * kernel_size_1)) % kernel_size_0)) + 
                128 * kernel_size_1 * kernel_size_1 * kernel_size_1 * (((temp_index_0 // ((-1) + ((-12) * kernel_size_1 * kernel_size_1) + 6 * kernel_size_1 + 8 * kernel_size_1 * kernel_size_1 * kernel_size_1)) % kernel_size_0)) + 
                ((temp_index_0 % kernel_size_2)) + 
                (((temp_index_0 // kernel_size_3) % kernel_size_2))
            ), 
            reduction_mask & temp_mask & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        temp_broadcast = tl.broadcast_to(temp_load, [XBLOCK, RBLOCK])
        temp_sum = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(reduction_mask & input_mask, temp_sum, temp_accumulator)

    temp_result = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (input_3), temp_result, input_mask)