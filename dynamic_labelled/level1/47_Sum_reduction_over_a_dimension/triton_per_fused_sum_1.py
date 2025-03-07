# From: 47_Sum_reduction_over_a_dimension

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_sum_1per_fused_sum_1(in_ptr0, out_ptr0, kernel_size, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 2
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    reduction_offset = reduction_index
    input_mod_kernel = (input_index % kernel_size)
    input_div_kernel = input_index // kernel_size
    original_index = input_index
    temp0 = tl.load(in_ptr0 + (input_mod_kernel + kernel_size * reduction_offset + 2 * kernel_size * input_div_kernel), input_mask, eviction_policy='evict_last', other=0.0)
    temp1 = tl.broadcast_to(temp0, [XBLOCK, RBLOCK])
    temp3 = tl.where(input_mask, temp1, 0)
    temp4 = tl.sum(temp3, 1)[:, None]
    tl.store(out_ptr0 + (original_index), temp4, input_mask)