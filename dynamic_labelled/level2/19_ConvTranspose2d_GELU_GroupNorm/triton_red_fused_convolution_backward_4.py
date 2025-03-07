# From: 19_ConvTranspose2d_GELU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_4(in_ptr0, out_ptr0, kernel_size, input_num_elements, output_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_num_elements = 384
    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < input_num_elements
    output_base = tl.arange(0, RBLOCK)[None, :]
    input_index_mod = input_indices % 64
    input_index_div = input_indices // 64
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_index_flat = input_indices

    for output_offset in range(0, output_num_elements, RBLOCK):
        output_indices = output_offset + output_base
        output_mask = output_indices < output_num_elements
        output_index_flat = output_indices
        temp_load = tl.load(
            in_ptr0 + (
                66 * (((output_index_flat + 726 * kernel_size * input_index_div) // 66) % 66) +
                4356 * input_index_mod +
                278784 * (((output_index_flat + 726 * kernel_size * input_index_div) // 4356) % kernel_size) +
                (output_index_flat % 66)
            ),
            output_mask & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        temp_broadcast = tl.broadcast_to(temp_load, [XBLOCK, RBLOCK])
        temp_sum = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(output_mask & input_mask, temp_sum, temp_accumulator)

    temp_result = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(out_ptr0 + (input_index_flat), temp_result, input_mask)