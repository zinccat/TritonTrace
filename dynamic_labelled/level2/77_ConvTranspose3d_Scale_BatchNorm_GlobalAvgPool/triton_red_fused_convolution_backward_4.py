# From: 77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_4red_fused_convolution_backward_4(
    input_ptr, output_ptr, kernel_size_z, kernel_size_y, kernel_size_x, input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 352
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_32 = input_index // 32
    input_index_mod_32 = input_index % 32
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_index_3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_2 = reduction_index

        temp_index_0 = reduction_index_2 + input_index_32 * (
            (10 + 4 * kernel_size_x * kernel_size_x + 8 * kernel_size_x + 
             kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
             2 * kernel_size_x * kernel_size_y * kernel_size_y + 
             4 * kernel_size_y * kernel_size_x * kernel_size_x + 
             8 * kernel_size_x * kernel_size_y) // 11
        )
        temp_index_1 = 4 * kernel_size_x * kernel_size_x + 8 * kernel_size_x + \
                       kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + \
                       2 * kernel_size_x * kernel_size_y * kernel_size_y + \
                       4 * kernel_size_y * kernel_size_x * kernel_size_x + \
                       8 * kernel_size_x * kernel_size_y
        temp_mask = temp_index_0 < temp_index_1

        temp_load = tl.load(
            input_ptr + (
                2 * (
                    (((reduction_index_2 + input_index_32 * (
                        (10 + 4 * kernel_size_x * kernel_size_x + 8 * kernel_size_x + 
                         kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
                         2 * kernel_size_x * kernel_size_y * kernel_size_y + 
                         4 * kernel_size_y * kernel_size_x * kernel_size_x + 
                         8 * kernel_size_x * kernel_size_y) // 11
                    )) // (2 + kernel_size_y)) % (2 + kernel_size_y)
                ) + 4 * (
                    (((reduction_index_2 + input_index_32 * (
                        (10 + 4 * kernel_size_x * kernel_size_x + 8 * kernel_size_x + 
                         kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
                         2 * kernel_size_x * kernel_size_y * kernel_size_y + 
                         4 * kernel_size_y * kernel_size_x * kernel_size_x + 
                         8 * kernel_size_x * kernel_size_y) // 11
                    )) // (4 + kernel_size_y * kernel_size_y + 4 * kernel_size_y)) % (2 + kernel_size_x)
                ) + 8 * input_index_mod_32 + 256 * (
                    (((reduction_index_2 + input_index_32 * (
                        (10 + 4 * kernel_size_x * kernel_size_x + 8 * kernel_size_x + 
                         kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
                         2 * kernel_size_x * kernel_size_y * kernel_size_y + 
                         4 * kernel_size_y * kernel_size_x * kernel_size_x + 
                         8 * kernel_size_x * kernel_size_y) // 11
                    )) // kernel_size_z) % kernel_size_x
                ) + kernel_size_y * (
                    (((reduction_index_2 + input_index_32 * (
                        (10 + 4 * kernel_size_x * kernel_size_x + 8 * kernel_size_x + 
                         kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
                         2 * kernel_size_x * kernel_size_y * kernel_size_y + 
                         4 * kernel_size_y * kernel_size_x * kernel_size_x + 
                         8 * kernel_size_x * kernel_size_y) // 11
                    )) // (2 + kernel_size_y)) % (2 + kernel_size_y)
                ) + kernel_size_y * kernel_size_y * (
                    (((reduction_index_2 + input_index_32 * (
                        (10 + 4 * kernel_size_x * kernel_size_x + 8 * kernel_size_x + 
                         kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
                         2 * kernel_size_x * kernel_size_y * kernel_size_y + 
                         4 * kernel_size_y * kernel_size_x * kernel_size_x + 
                         8 * kernel_size_x * kernel_size_y) // 11
                    )) // (4 + kernel_size_y * kernel_size_y + 4 * kernel_size_y)) % (2 + kernel_size_x)
                ) + 2 * input_index_mod_32 * kernel_size_y * kernel_size_y + 4 * kernel_size_x * input_index_mod_32 + 
                4 * kernel_size_y * (
                    (((reduction_index_2 + input_index_32 * (
                        (10 + 4 * kernel_size_x * kernel_size_x + 8 * kernel_size_x + 
                         kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
                         2 * kernel_size_x * kernel_size_y * kernel_size_y + 
                         4 * kernel_size_y * kernel_size_x * kernel_size_x + 
                         8 * kernel_size_x * kernel_size_y) // 11
                    )) // (4 + kernel_size_y * kernel_size_y + 4 * kernel_size_y)) % (2 + kernel_size_x)
                ) + 8 * kernel_size_y * input_index_mod_32 + 64 * kernel_size_y * kernel_size_y * (
                    (((reduction_index_2 + input_index_32 * (
                        (10 + 4 * kernel_size_x * kernel_size_x + 8 * kernel_size_x + 
                         kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
                         2 * kernel_size_x * kernel_size_y * kernel_size_y + 
                         4 * kernel_size_y * kernel_size_x * kernel_size_x + 
                         8 * kernel_size_x * kernel_size_y) // 11
                    )) // kernel_size_z) % kernel_size_x
                ) + 128 * kernel_size_x * (
                    (((reduction_index_2 + input_index_32 * (
                        (10 + 4 * kernel_size_x * kernel_size_x + 8 * kernel_size_x + 
                         kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
                         2 * kernel_size_x * kernel_size_y * kernel_size_y + 
                         4 * kernel_size_y * kernel_size_x * kernel_size_x + 
                         8 * kernel_size_x * kernel_size_y) // 11
                    )) // kernel_size_z) % kernel_size_x
                ) + 256 * kernel_size_y * (
                    (((reduction_index_2 + input_index_32 * (
                        (10 + 4 * kernel_size_x * kernel_size_x + 8 * kernel_size_x + 
                         kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
                         2 * kernel_size_x * kernel_size_y * kernel_size_y + 
                         4 * kernel_size_y * kernel_size_x * kernel_size_x + 
                         8 * kernel_size_x * kernel_size_y) // 11
                    )) // kernel_size_z) % kernel_size_x
                ) + kernel_size_x * input_index_mod_32 * kernel_size_y * kernel_size_y + 
                4 * kernel_size_x * kernel_size_y * input_index_mod_32 + 32 * kernel_size_x * kernel_size_y * kernel_size_y * (
                    (((reduction_index_2 + input_index_32 * (
                        (10 + 4 * kernel_size_x * kernel_size_x + 8 * kernel_size_x + 
                         kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
                         2 * kernel_size_x * kernel_size_y * kernel_size_y + 
                         4 * kernel_size_y * kernel_size_x * kernel_size_x + 
                         8 * kernel_size_x * kernel_size_y) // 11
                    )) // kernel_size_z) % kernel_size_x
                ) + 128 * kernel_size_x * kernel_size_y * (
                    (((reduction_index_2 + input_index_32 * (
                        (10 + 4 * kernel_size_x * kernel_size_x + 8 * kernel_size_x + 
                         kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
                         2 * kernel_size_x * kernel_size_y * kernel_size_y + 
                         4 * kernel_size_y * kernel_size_x * kernel_size_x + 
                         8 * kernel_size_x * kernel_size_y) // 11
                    )) // kernel_size_z) % kernel_size_x
                ) + (
                    (((reduction_index_2 + input_index_32 * (
                        (10 + 4 * kernel_size_x * kernel_size_x + 8 * kernel_size_x + 
                         kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
                         2 * kernel_size_x * kernel_size_y * kernel_size_y + 
                         4 * kernel_size_y * kernel_size_x * kernel_size_x + 
                         8 * kernel_size_x * kernel_size_y) // 11
                    )) % (2 + kernel_size_y))
                )
            ), reduction_mask & temp_mask & input_mask, eviction_policy='evict_last', other=0.0
        )

        temp_broadcast = tl.broadcast_to(temp_load, [XBLOCK, RBLOCK])
        temp_accumulator_update = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(reduction_mask & input_mask, temp_accumulator_update, temp_accumulator)

    temp_sum = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (input_index_3), temp_sum, input_mask)