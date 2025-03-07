# From: 36_RMSNorm_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mean_pow_0red_fused_mean_pow_0(input_ptr, output_ptr, kernel_size_0, kernel_size_1, kernel_size_2, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_x0 = (input_index % kernel_size_0)
    input_x1 = input_index // kernel_size_0
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_x3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_r2 = reduction_index
        temp0 = tl.load(input_ptr + (input_x0 + reduction_r2 * kernel_size_2 * kernel_size_2 + kernel_size_1 * input_x1 * kernel_size_2 * kernel_size_2), reduction_mask & input_mask, eviction_policy='evict_last', other=0.0)
        temp1 = temp0 * temp0
        temp2 = tl.broadcast_to(temp1, [XBLOCK, RBLOCK])
        temp4 = temp_accumulator + temp2
        temp_accumulator = tl.where(reduction_mask & input_mask, temp4, temp_accumulator)

    temp3 = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (input_x3), temp3, input_mask)