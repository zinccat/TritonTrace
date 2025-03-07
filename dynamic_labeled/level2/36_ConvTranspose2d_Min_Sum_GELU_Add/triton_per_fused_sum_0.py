# From: 36_ConvTranspose2d_Min_Sum_GELU_Add

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_sum_0(input_ptr, output_ptr, kernel_size_0, kernel_size_1, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 16
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    reduction_index_2 = reduction_index
    kernel_index_0 = (input_index % kernel_size_0)
    kernel_index_1 = input_index // kernel_size_0
    linear_index = input_index
    temp0 = tl.load(input_ptr + (kernel_index_0 + 2 * kernel_size_1 * reduction_index_2 + 32 * kernel_size_1 * kernel_index_1), input_mask, eviction_policy='evict_last', other=0.0)
    temp1 = tl.broadcast_to(temp0, [XBLOCK, RBLOCK])
    temp3 = tl.where(input_mask, temp1, 0)
    temp4 = tl.sum(temp3, 1)[:, None]
    tl.store(output_ptr + (linear_index), temp4, input_mask)