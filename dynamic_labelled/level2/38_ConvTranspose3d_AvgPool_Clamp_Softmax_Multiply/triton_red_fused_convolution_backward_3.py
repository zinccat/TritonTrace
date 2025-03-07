# From: 38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_3(input_ptr, output_ptr, kernel_size_z, kernel_size_y, kernel_size_x, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
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
        reduction_z = reduction_index

        temp_index = reduction_z + input_x * ((20 + 8 * kernel_size_z * kernel_size_z * kernel_size_y * kernel_size_y) // 21)
        temp_kernel_product = 8 * kernel_size_z * kernel_size_z * kernel_size_y * kernel_size_y
        temp_condition = temp_index < temp_kernel_product

        temp_load = tl.load(
            input_ptr + (
                8 * kernel_size_z * input_y * kernel_size_y * kernel_size_y +
                128 * kernel_size_z * kernel_size_y * kernel_size_y * (
                    ((temp_index // kernel_size_x) % kernel_size_z) +
                    ((temp_index % kernel_size_x))
                )
            ),
            reduction_mask & temp_condition & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        temp_broadcast = tl.broadcast_to(temp_load, [XBLOCK, RBLOCK])
        temp_sum = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(reduction_mask & input_mask, temp_sum, temp_accumulator)

    temp_result = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (input_flat_index), temp_result, input_mask)