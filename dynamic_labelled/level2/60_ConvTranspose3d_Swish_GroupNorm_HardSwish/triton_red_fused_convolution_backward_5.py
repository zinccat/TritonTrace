# From: 60_ConvTranspose3d_Swish_GroupNorm_HardSwish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_5(
    input_ptr, output_ptr, kernel_size_0, kernel_size_1, kernel_size_2, kernel_size_3, kernel_size_4, kernel_size_5,
    x_num_elements, r_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_num_elements = 1968
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < x_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_div_16 = x_indices // 16
    x_mod_16 = x_indices % 16
    temp_buffer = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x_indices_flat = x_indices

    for r_offset in range(0, r_num_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < r_num_elements
        r_indices_flat = r_indices

        divisor = triton_helpers.div_floor_integer(
            122 + ((-1) * kernel_size_0) + ((-4) * kernel_size_0 * kernel_size_2 * kernel_size_2) +
            2 * kernel_size_0 * kernel_size_1 + 4 * kernel_size_0 * kernel_size_2 +
            ((-8) * kernel_size_0 * kernel_size_1 * kernel_size_2) + 8 * kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2,
            123
        )

        temp_index_0 = r_indices_flat + x_div_16 * divisor
        temp_index_1 = ((-1) * kernel_size_0) + ((-4) * kernel_size_0 * kernel_size_2 * kernel_size_2) + \
                       2 * kernel_size_0 * kernel_size_1 + 4 * kernel_size_0 * kernel_size_2 + \
                       ((-8) * kernel_size_0 * kernel_size_1 * kernel_size_2) + 8 * kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2
        temp_mask = temp_index_0 < temp_index_1

        load_index = (((-1) * x_mod_16) + ((-1) * (((temp_index_0 // kernel_size_3) % kernel_size_3))) +
                      ((-16) * (((temp_index_0 // kernel_size_5) % kernel_size_0))) +
                      ((-64) * kernel_size_2 * kernel_size_2 * (((temp_index_0 // kernel_size_5) % kernel_size_0))) +
                      ((-4) * kernel_size_2 * (((temp_index_0 // (1 + ((-4) * kernel_size_2) + 4 * kernel_size_2 * kernel_size_2)) % kernel_size_4))) +
                      ((-4) * x_mod_16 * kernel_size_2 * kernel_size_2) + 2 * kernel_size_1 * x_mod_16 +
                      2 * kernel_size_2 * (((temp_index_0 // kernel_size_3) % kernel_size_3)) +
                      4 * kernel_size_2 * x_mod_16 + 4 * kernel_size_2 * kernel_size_2 * (((temp_index_0 // (1 + ((-4) * kernel_size_2) + 4 * kernel_size_2 * kernel_size_2)) % kernel_size_4)) +
                      32 * kernel_size_1 * (((temp_index_0 // kernel_size_5) % kernel_size_0)) +
                      64 * kernel_size_2 * (((temp_index_0 // kernel_size_5) % kernel_size_0)) +
                      ((-128) * kernel_size_1 * kernel_size_2 * (((temp_index_0 // kernel_size_5) % kernel_size_0))) +
                      ((-8) * kernel_size_1 * kernel_size_2 * x_mod_16) +
                      8 * kernel_size_1 * x_mod_16 * kernel_size_2 * kernel_size_2 +
                      128 * kernel_size_1 * kernel_size_2 * kernel_size_2 * (((temp_index_0 // kernel_size_5) % kernel_size_0)) +
                      (((temp_index_0 % kernel_size_3)) + (((temp_index_0 // (1 + ((-4) * kernel_size_2) + 4 * kernel_size_2 * kernel_size_2)) % kernel_size_4))))

        temp_data = tl.load(input_ptr + load_index, rmask & temp_mask & x_mask, eviction_policy='evict_last', other=0.0)
        temp_broadcast = tl.broadcast_to(temp_data, [XBLOCK, RBLOCK])
        temp_accumulate = temp_buffer + temp_broadcast
        temp_buffer = tl.where(rmask & x_mask, temp_accumulate, temp_buffer)

    temp_result = tl.sum(temp_buffer, 1)[:, None]
    tl.store(output_ptr + (x_indices_flat), temp_result, x_mask)