# From: 26_ConvTranspose3d_Add_HardSwish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_1(in_ptr0, out_ptr0, kernel_size, input_num_elements, output_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_num_elements = 2048
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    output_base = tl.arange(0, RBLOCK)[None, :]
    input_x0 = (input_index % 64)
    input_x1 = input_index // 64
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_x3 = input_index

    for output_offset in range(0, output_num_elements, RBLOCK):
        output_index = output_offset + output_base
        output_mask = output_index < output_num_elements
        output_r2 = output_index
        temp0 = tl.load(
            in_ptr0 + (
                1024 * (((output_r2 + 1024 * kernel_size * input_x1) // 1024) % 32) +
                32768 * input_x0 +
                2097152 * (((output_r2 + 1024 * kernel_size * input_x1) // 32768) % kernel_size) +
                (output_r2 % 1024)
            ),
            output_mask & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        temp1 = tl.broadcast_to(temp0, [XBLOCK, RBLOCK])
        temp3 = temp_accumulator + temp1
        temp_accumulator = tl.where(output_mask & input_mask, temp3, temp_accumulator)

    temp2 = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(out_ptr0 + (input_x3), temp2, input_mask)