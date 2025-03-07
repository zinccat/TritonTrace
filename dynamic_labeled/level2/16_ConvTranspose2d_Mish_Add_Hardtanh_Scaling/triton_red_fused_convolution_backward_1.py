# From: 16_ConvTranspose2d_Mish_Add_Hardtanh_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_1(input_ptr, output_ptr, kernel_size_0, kernel_size_1, input_num_elements, output_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_num_elements = 448
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    output_base = tl.arange(0, RBLOCK)[None, :]
    input_block_1 = input_index // 64
    input_block_0 = (input_index % 64)
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_block_3 = input_index

    for output_offset in range(0, output_num_elements, RBLOCK):
        output_index = output_offset + output_base
        output_mask = output_index < output_num_elements
        output_block_2 = output_index

        temp_index_0 = output_block_2 + input_block_1 * ((6 + kernel_size_0 + 4 * kernel_size_0 * kernel_size_1 + 4 * kernel_size_0 * kernel_size_1 * kernel_size_1) // 7)
        temp_index_1 = kernel_size_0 + 4 * kernel_size_0 * kernel_size_1 + 4 * kernel_size_0 * kernel_size_1 * kernel_size_1
        temp_mask_2 = temp_index_0 < temp_index_1

        temp_load = tl.load(
            input_ptr + (
                input_block_0 + 64 * (
                    ((temp_index_0 // (1 + 4 * kernel_size_1 + 4 * kernel_size_1 * kernel_size_1)) % kernel_size_0)
                ) + 2 * kernel_size_1 * (
                    ((temp_index_0 // (1 + 2 * kernel_size_1)) % (1 + 2 * kernel_size_1))
                ) + 4 * kernel_size_1 * input_block_0 + 4 * input_block_0 * kernel_size_1 * kernel_size_1 + 256 * kernel_size_1 * (
                    ((temp_index_0 // (1 + 4 * kernel_size_1 + 4 * kernel_size_1 * kernel_size_1)) % kernel_size_0)
                ) + 256 * kernel_size_1 * kernel_size_1 * (
                    ((temp_index_0 // (1 + 4 * kernel_size_1 + 4 * kernel_size_1 * kernel_size_1)) % kernel_size_0)
                ) + ((temp_index_0 % (1 + 2 * kernel_size_1))) + (
                    (((temp_index_0 // (1 + 2 * kernel_size_1)) % (1 + 2 * kernel_size_1)))
                )
            ),
            output_mask & temp_mask_2 & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        temp_broadcast = tl.broadcast_to(temp_load, [XBLOCK, RBLOCK])
        temp_accumulate = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(output_mask & input_mask, temp_accumulate, temp_accumulator)

    temp_sum = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (input_block_3), temp_sum, input_mask)