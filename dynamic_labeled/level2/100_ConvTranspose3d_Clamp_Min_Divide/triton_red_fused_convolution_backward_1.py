# From: 100_ConvTranspose3d_Clamp_Min_Divide

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_1(in_ptr0, out_ptr0, kernel_size, input_num_elements, output_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_num_elements = 336
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    output_base = tl.arange(0, RBLOCK)[None, :]
    input_x = (input_index % 21)
    input_y = input_index // 21
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_flat_index = input_index

    for output_offset in range(0, output_num_elements, RBLOCK):
        output_index = output_offset + output_base
        output_mask = output_index < output_num_elements
        output_flat_index = output_index

        temp_load = tl.load(
            in_ptr0 + (
                (-63504) * (
                    (((output_flat_index + ((-189) * kernel_size * input_x) + 378 * input_x * kernel_size * kernel_size) // ((-3969) + 7938 * kernel_size)) % kernel_size)
                ) + 
                ((-3969) * input_y) + 
                63 * (
                    (((output_flat_index + ((-189) * kernel_size * input_x) + 378 * input_x * kernel_size * kernel_size) // 63) % ((-63) + 126 * kernel_size))
                ) + 
                7938 * kernel_size * input_y + 
                127008 * kernel_size * (
                    (((output_flat_index + ((-189) * kernel_size * input_x) + 378 * input_x * kernel_size * kernel_size) // ((-3969) + 7938 * kernel_size)) % kernel_size)
                ) + 
                (output_flat_index % 63)
            ), 
            output_mask & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        temp_broadcast = tl.broadcast_to(temp_load, [XBLOCK, RBLOCK])
        temp_sum = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(output_mask & input_mask, temp_sum, temp_accumulator)

    temp_result = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(out_ptr0 + (input_flat_index), temp_result, input_mask)