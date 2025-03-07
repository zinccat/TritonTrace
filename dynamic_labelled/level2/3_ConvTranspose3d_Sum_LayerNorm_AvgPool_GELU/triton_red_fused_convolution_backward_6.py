# From: 3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_6(
    input_ptr, output_ptr, kernel_size_0, kernel_size_1, kernel_size_2, 
    input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, 
    RBLOCK: tl.constexpr
):
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    reduction_mask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_x0 = (input_index % 64)
    input_x1 = input_index // 64
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_x3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_r2 = reduction_index
        temp_input = tl.load(
            input_ptr + (
                64 * (((reduction_r2 + 64 * kernel_size_1 * kernel_size_2 * input_x1) // 64) % kernel_size_0)
                + 8192 * kernel_size_2 * input_x0
                + 524288 * kernel_size_2 * (((reduction_r2 + 64 * kernel_size_1 * kernel_size_2 * input_x1) // (8192 * kernel_size_2)) % kernel_size_1)
                + (reduction_r2 % 64)
            ),
            reduction_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        temp_broadcast = tl.broadcast_to(temp_input, [XBLOCK, RBLOCK])
        temp_sum = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(reduction_mask, temp_sum, temp_accumulator)

    temp_result = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (input_x3), temp_result, None)