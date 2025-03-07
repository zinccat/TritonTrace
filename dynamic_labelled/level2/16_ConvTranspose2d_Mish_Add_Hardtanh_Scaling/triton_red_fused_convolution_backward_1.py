# From: 16_ConvTranspose2d_Mish_Add_Hardtanh_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_1red_fused_convolution_backward_1(
    input_ptr, output_ptr, kernel_size_0, kernel_size_1, input_num_elements, result_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 448
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    result_base = tl.arange(0, RBLOCK)[None, :]
    input_index_1 = input_index // 64
    input_index_0 = (input_index % 64)
    temp_result = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_index_3 = input_index

    for result_offset in range(0, result_num_elements, RBLOCK):
        result_index = result_offset + result_base
        result_mask = result_index < result_num_elements
        result_index_2 = result_index

        temp0 = result_index_2 + input_index_1 * ((6 + kernel_size_0 + 4 * kernel_size_0 * kernel_size_1 + 4 * kernel_size_0 * kernel_size_1 * kernel_size_1) // 7)
        temp1 = kernel_size_0 + 4 * kernel_size_0 * kernel_size_1 + 4 * kernel_size_0 * kernel_size_1 * kernel_size_1
        temp2 = temp0 < temp1

        input_load_index = (
            input_index_0 + 64 * (
                ((result_index_2 + input_index_1 * ((6 + kernel_size_0 + 4 * kernel_size_0 * kernel_size_1 + 4 * kernel_size_0 * kernel_size_1 * kernel_size_1) // 7)) // (1 + 4 * kernel_size_1 + 4 * kernel_size_1 * kernel_size_1)) % kernel_size_0
            ) + 2 * kernel_size_1 * (
                ((result_index_2 + input_index_1 * ((6 + kernel_size_0 + 4 * kernel_size_0 * kernel_size_1 + 4 * kernel_size_0 * kernel_size_1 * kernel_size_1) // 7)) // (1 + 2 * kernel_size_1)) % (1 + 2 * kernel_size_1)
            ) + 4 * kernel_size_1 * input_index_0 + 4 * input_index_0 * kernel_size_1 * kernel_size_1 + 256 * kernel_size_1 * (
                ((result_index_2 + input_index_1 * ((6 + kernel_size_0 + 4 * kernel_size_0 * kernel_size_1 + 4 * kernel_size_0 * kernel_size_1 * kernel_size_1) // 7)) // (1 + 4 * kernel_size_1 + 4 * kernel_size_1 * kernel_size_1)) % kernel_size_0
            ) + 256 * kernel_size_1 * kernel_size_1 * (
                ((result_index_2 + input_index_1 * ((6 + kernel_size_0 + 4 * kernel_size_0 * kernel_size_1 + 4 * kernel_size_0 * kernel_size_1 * kernel_size_1) // 7)) // (1 + 4 * kernel_size_1 + 4 * kernel_size_1 * kernel_size_1)) % kernel_size_0
            ) + (
                (result_index_2 + input_index_1 * ((6 + kernel_size_0 + 4 * kernel_size_0 * kernel_size_1 + 4 * kernel_size_0 * kernel_size_1 * kernel_size_1) // 7)) % (1 + 2 * kernel_size_1)
            ) + (
                ((result_index_2 + input_index_1 * ((6 + kernel_size_0 + 4 * kernel_size_0 * kernel_size_1 + 4 * kernel_size_0 * kernel_size_1 * kernel_size_1) // 7)) // (1 + 2 * kernel_size_1)) % (1 + 2 * kernel_size_1)
            )
        )

        temp3 = tl.load(input_ptr + input_load_index, result_mask & temp2 & input_mask, eviction_policy='evict_last', other=0.0)
        temp4 = tl.broadcast_to(temp3, [XBLOCK, RBLOCK])
        temp6 = temp_result + temp4
        temp_result = tl.where(result_mask & input_mask, temp6, temp_result)

    temp5 = tl.sum(temp_result, 1)[:, None]
    tl.store(output_ptr + (input_index_3), temp5, input_mask)