# From: 20_ConvTranspose3d_Sum_ResidualAdd_Multiply_ResidualAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_sum_1(in_ptr0, out_ptr0, out_ptr1, kernel_size, input_num_elements, residual_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_num_elements = 1024
    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < input_num_elements
    residual_base = tl.arange(0, RBLOCK)[None, :]
    input_index_mod_64 = (input_indices % 64)
    input_index_div_64 = input_indices // 64
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_index = input_indices

    for residual_offset in range(0, residual_num_elements, RBLOCK):
        residual_indices = residual_offset + residual_base
        residual_mask = residual_indices < residual_num_elements
        residual_index = residual_indices

        temp_load = tl.load(
            in_ptr0 + (
                64 * (((residual_index + 512 * input_index_div_64 * kernel_size * kernel_size) // 64) % (128 * kernel_size))
                + 8192 * kernel_size * input_index_mod_64
                + 524288 * kernel_size * (((residual_index + 512 * input_index_div_64 * kernel_size * kernel_size) // (8192 * kernel_size)) % kernel_size)
                + (residual_index % 64)
            ),
            residual_mask & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        temp_broadcast = tl.broadcast_to(temp_load, [XBLOCK, RBLOCK])
        temp_sum = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(residual_mask & input_mask, temp_sum, temp_accumulator)

    temp_result = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(out_ptr0 + (input_index), temp_result, input_mask)
    tl.store(out_ptr1 + (input_index), temp_result, input_mask)