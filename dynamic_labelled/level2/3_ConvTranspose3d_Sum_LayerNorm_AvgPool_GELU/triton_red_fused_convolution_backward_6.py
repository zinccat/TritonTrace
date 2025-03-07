# From: 3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_6(input_ptr, output_ptr, kernel_size_0, kernel_size_1, kernel_size_2, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_dim_0 = (input_index % 64)
    input_dim_1 = input_index // 64
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_flat_index = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_flat_index = reduction_index
        temp_load = tl.load(
            input_ptr + (
                64 * (((reduction_flat_index + 64 * kernel_size_1 * kernel_size_2 * input_dim_1) // 64) % kernel_size_0)
                + 8192 * kernel_size_2 * input_dim_0
                + 524288 * kernel_size_2 * (((reduction_flat_index + 64 * kernel_size_1 * kernel_size_2 * input_dim_1) // (8192 * kernel_size_2)) % kernel_size_1)
                + (reduction_flat_index % 64)
            ),
            reduction_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        temp_broadcast = tl.broadcast_to(temp_load, [XBLOCK, RBLOCK])
        temp_sum = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(reduction_mask, temp_sum, temp_accumulator)

    temp_result = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (input_flat_index), temp_result, None)