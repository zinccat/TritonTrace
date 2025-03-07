# From: 23_Conv3d_GroupNorm_Mean

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_4red_fused_convolution_backward_4(
    input_ptr, output_ptr, kernel_size_0, kernel_size_1, kernel_size_2, kernel_size_3, 
    input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 336
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_1 = input_index // 16
    input_0 = (input_index % 16)
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_2 = reduction_index

        divisor = triton_helpers.div_floor_integer(
            20 + ((-8) * kernel_size_0) + ((-2) * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
            4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
            kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_2), 
            21
        )

        temp_index_0 = reduction_2 + input_1 * divisor
        temp_index_1 = ((-8) * kernel_size_0) + ((-2) * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                       4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                       kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_2)

        index_mask = temp_index_0 < temp_index_1

        load_index = (
            ((-128) * (((temp_index_0 // kernel_size_3) % kernel_size_0))) + 
            ((-8) * input_0) + 
            ((-2) * (((temp_index_0 // ((-2) + kernel_size_2)) % ((-2) + kernel_size_2)))) + 
            4 * (((temp_index_0 // (4 + kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_2))) % ((-2) + kernel_size_1))) + 
            kernel_size_2 * (((temp_index_0 // ((-2) + kernel_size_2)) % ((-2) + kernel_size_2))) + 
            kernel_size_2 * kernel_size_2 * (((temp_index_0 // (4 + kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_2))) % ((-2) + kernel_size_1))) + 
            ((-32) * kernel_size_2 * kernel_size_2 * (((temp_index_0 // kernel_size_3) % kernel_size_0))) + 
            ((-4) * kernel_size_2 * (((temp_index_0 // (4 + kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_2))) % ((-2) + kernel_size_1)))) + 
            ((-2) * input_0 * kernel_size_2 * kernel_size_2) + 
            4 * kernel_size_1 * input_0 + 
            8 * kernel_size_2 * input_0 + 
            64 * kernel_size_1 * (((temp_index_0 // kernel_size_3) % kernel_size_0)) + 
            128 * kernel_size_2 * (((temp_index_0 // kernel_size_3) % kernel_size_0)) + 
            kernel_size_1 * input_0 * kernel_size_2 * kernel_size_2 + 
            ((-64) * kernel_size_1 * kernel_size_2 * (((temp_index_0 // kernel_size_3) % kernel_size_0))) + 
            ((-4) * kernel_size_1 * kernel_size_2 * input_0) + 
            16 * kernel_size_1 * kernel_size_2 * kernel_size_2 * (((temp_index_0 // kernel_size_3) % kernel_size_0)) + 
            (((temp_index_0 % ((-2) + kernel_size_2))))
        )

        temp_data = tl.load(
            input_ptr + load_index, 
            mask=reduction_mask & index_mask & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        temp_broadcast = tl.broadcast_to(temp_data, [XBLOCK, RBLOCK])
        temp_accumulator += temp_broadcast
        temp_accumulator = tl.where(reduction_mask & input_mask, temp_accumulator, temp_accumulator)

    result_sum = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (input_3), result_sum, input_mask)