# From: 15_ConvTranspose3d_BatchNorm_Subtract

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_5(input_ptr, output_ptr, kernel_size_0, kernel_size_1, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_num_elements = 352
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_div_32 = input_index // 32
    input_mod_32 = input_index % 32
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_flat_index = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_flat_index = reduction_index

        temp_index_0 = reduction_flat_index + 46 * input_div_32 + input_div_32 * (triton_helpers.div_floor_integer((-1984 * kernel_size_0) + 1984 * kernel_size_0 * kernel_size_0, 11))
        temp_index_1 = 496 + ((-1984) * kernel_size_0) + 1984 * kernel_size_0 * kernel_size_0
        temp_mask = temp_index_0 < temp_index_1

        temp_load_index = (
            -1 * (((temp_index_0 // ((-1) + 2 * kernel_size_0)) % ((-1) + 2 * kernel_size_0))) +
            31 * input_mod_32 +
            992 * (((temp_index_0 // kernel_size_1) % 16)) +
            (-3968) * kernel_size_0 * (((temp_index_0 // kernel_size_1) % 16)) +
            (-124) * kernel_size_0 * input_mod_32 +
            (-4) * kernel_size_0 * (((temp_index_0 // (1 + ((-4) * kernel_size_0) + 4 * kernel_size_0 * kernel_size_0)) % 31)) +
            2 * kernel_size_0 * (((temp_index_0 // ((-1) + 2 * kernel_size_0)) % ((-1) + 2 * kernel_size_0))) +
            4 * kernel_size_0 * kernel_size_0 * (((temp_index_0 // (1 + ((-4) * kernel_size_0) + 4 * kernel_size_0 * kernel_size_0)) % 31)) +
            124 * input_mod_32 * kernel_size_0 * kernel_size_0 +
            3968 * kernel_size_0 * kernel_size_0 * (((temp_index_0 // kernel_size_1) % 16)) +
            (temp_index_0 % ((-1) + 2 * kernel_size_0)) +
            (((temp_index_0 // (1 + ((-4) * kernel_size_0) + 4 * kernel_size_0 * kernel_size_0)) % 31))
        )

        temp_load = tl.load(input_ptr + temp_load_index, reduction_mask & temp_mask & input_mask, eviction_policy='evict_last', other=0.0)
        temp_broadcast = tl.broadcast_to(temp_load, [XBLOCK, RBLOCK])
        temp_accumulated = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(reduction_mask & input_mask, temp_accumulated, temp_accumulator)

    temp_sum = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (input_flat_index), temp_sum, input_mask)