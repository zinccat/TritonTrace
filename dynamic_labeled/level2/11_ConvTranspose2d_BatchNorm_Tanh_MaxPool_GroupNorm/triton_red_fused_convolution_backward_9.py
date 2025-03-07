# From: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_9(input_ptr, output_ptr, kernel_size, input_num_elements, output_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_num_elements = 384
    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < input_num_elements
    output_base = tl.arange(0, RBLOCK)[None, :]
    input_index_64 = input_indices // 64
    input_index_mod = input_indices % 64
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_index = input_indices

    for output_offset in range(0, output_num_elements, RBLOCK):
        output_indices = output_offset + output_base
        output_mask = output_indices < output_num_elements
        output_index = output_indices
        temp_index = output_index + input_index_64 * ((5 + 4096 * kernel_size) // 6)
        max_temp_index = 4096 * kernel_size
        valid_temp_index = temp_index < max_temp_index
        temp_load = tl.load(
            input_ptr + (4096 * input_index_mod + 262144 * (((temp_index // 4096) % kernel_size)) + (temp_index % 4096)),
            valid_temp_index & output_mask & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        temp_broadcast = tl.broadcast_to(temp_load, [XBLOCK, RBLOCK])
        temp_accumulator += temp_broadcast
        temp_accumulator = tl.where(output_mask & input_mask, temp_accumulator, temp_accumulator)

    temp_sum = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (input_index), temp_sum, input_mask)