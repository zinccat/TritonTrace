# From: 15_ConvTranspose3d_BatchNorm_Subtract

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_5(
    input_ptr, output_ptr, kernel_size_0, kernel_size_1, input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 352
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_1 = input_index // 32
    input_index_0 = (input_index % 32)
    temp_buffer = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_index_3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_2 = reduction_index

        temp_index_0 = reduction_index_2 + 46 * input_index_1 + input_index_1 * (
            triton_helpers.div_floor_integer((-1984 * kernel_size_0) + 1984 * kernel_size_0 * kernel_size_0, 11)
        )
        temp_index_1 = 496 + ((-1984) * kernel_size_0) + 1984 * kernel_size_0 * kernel_size_0
        temp_index_2 = temp_index_0 < temp_index_1

        load_index = (
            -1 * (((temp_index_0 // ((-1) + 2 * kernel_size_0)) % ((-1) + 2 * kernel_size_0))) 
            + 31 * input_index_0 
            + 992 * (((temp_index_0 // kernel_size_1) % 16)) 
            + (-3968) * kernel_size_0 * (((temp_index_0 // kernel_size_1) % 16)) 
            + (-124) * kernel_size_0 * input_index_0 
            + (-4) * kernel_size_0 * (((temp_index_0 // (1 + ((-4) * kernel_size_0) + 4 * kernel_size_0 * kernel_size_0)) % 31)) 
            + 2 * kernel_size_0 * (((temp_index_0 // ((-1) + 2 * kernel_size_0)) % ((-1) + 2 * kernel_size_0))) 
            + 4 * kernel_size_0 * kernel_size_0 * (((temp_index_0 // (1 + ((-4) * kernel_size_0) + 4 * kernel_size_0 * kernel_size_0)) % 31)) 
            + 124 * input_index_0 * kernel_size_0 * kernel_size_0 
            + 3968 * kernel_size_0 * kernel_size_0 * (((temp_index_0 // kernel_size_1) % 16)) 
            + ((temp_index_0 % ((-1) + 2 * kernel_size_0))) 
            + (((temp_index_0 // (1 + ((-4) * kernel_size_0) + 4 * kernel_size_0 * kernel_size_0)) % 31))
        )

        temp_data = tl.load(
            input_ptr + load_index, 
            reduction_mask & temp_index_2 & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        broadcasted_data = tl.broadcast_to(temp_data, [XBLOCK, RBLOCK])
        temp_buffer_updated = temp_buffer + broadcasted_data
        temp_buffer = tl.where(reduction_mask & input_mask, temp_buffer_updated, temp_buffer)

    result = tl.sum(temp_buffer, 1)[:, None]
    tl.store(output_ptr + (input_index_3), result, input_mask)