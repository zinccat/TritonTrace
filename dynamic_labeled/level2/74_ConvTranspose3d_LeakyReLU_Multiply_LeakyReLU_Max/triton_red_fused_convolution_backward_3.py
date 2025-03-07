# From: 74_ConvTranspose3d_LeakyReLU_Multiply_LeakyReLU_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_3(in_ptr0, out_ptr0, kernel_size0, kernel_size1, input_num_elements, output_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_num_elements = 352
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    output_base = tl.arange(0, RBLOCK)[None, :]
    input_index_div_32 = input_index // 32
    input_index_mod_32 = input_index % 32
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_index_flat = input_index

    for output_offset in range(0, output_num_elements, RBLOCK):
        output_index = output_offset + output_base
        output_mask = output_index < output_num_elements
        output_index_flat = output_index
        temp_index = output_index_flat + input_index_div_32 * ((10 + 2048 * kernel_size0 * kernel_size0) // 11)
        max_temp_index = 2048 * kernel_size0 * kernel_size0
        index_within_bounds = temp_index < max_temp_index

        temp_load = tl.load(
            in_ptr0 + (
                128 * input_index_mod_32 * kernel_size0 * kernel_size0 +
                4096 * kernel_size0 * kernel_size0 * (
                    ((temp_index // kernel_size1) % 16) +
                    (temp_index % kernel_size1)
                )
            ),
            output_mask & index_within_bounds & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        temp_broadcast = tl.broadcast_to(temp_load, [XBLOCK, RBLOCK])
        temp_accumulator += temp_broadcast
        temp_accumulator = tl.where(output_mask & input_mask, temp_accumulator, temp_accumulator)

    temp_sum = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(out_ptr0 + (input_index_flat), temp_sum, input_mask)