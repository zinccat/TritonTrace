# From: 24_Conv3d_Min_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_1per_fused__softmax_1(in_ptr0, out_ptr0, out_ptr1, kernel_size0, kernel_size1, total_elements, row_elements, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 16
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:, None]
    mask = index < total_elements
    row_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    row_offset = row_index
    kernel_index = (index % kernel_size0)
    batch_index = index // kernel_size0
    flat_index = index
    input_value = tl.load(
        in_ptr0 + (kernel_index + 4*row_offset + 64*batch_index + row_offset*kernel_size1*kernel_size1 + (-64)*kernel_size1*batch_index + (-4)*kernel_size1*row_offset + 16*batch_index*kernel_size1*kernel_size1),
        mask,
        eviction_policy='evict_last',
        other=0.0
    )
    broadcasted_input = tl.broadcast_to(input_value, [XBLOCK, RBLOCK])
    masked_input = tl.where(mask, broadcasted_input, float("-inf"))
    max_value = triton_helpers.max2(masked_input, 1)[:, None]
    shifted_input = input_value - max_value
    exp_input = tl.math.exp(shifted_input)
    broadcasted_exp = tl.broadcast_to(exp_input, [XBLOCK, RBLOCK])
    masked_exp = tl.where(mask, broadcasted_exp, 0)
    sum_exp = tl.sum(masked_exp, 1)[:, None]
    tl.store(out_ptr0 + (flat_index), max_value, mask)
    tl.store(out_ptr1 + (flat_index), sum_exp, mask)