# From: 6_Conv3d_Softmax_MaxPool_MaxPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused__softmax_convolution_0(input_ptr0, input_ptr1, output_ptr0, output_ptr1, total_elements, reduction_elements, XBLOCK: tl.constexpr):
    total_elements = 1612800
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_indices
    x0 = x_indices % 12600
    x1 = (x_indices // 12600)
    
    input_value0 = tl.load(input_ptr0 + (x0 + (12600 * r2) + (201600 * x1)), x_mask, other=0.0)
    input_value1 = tl.load(input_ptr1 + (r2), None, eviction_policy='evict_last')
    combined_values = input_value0 + input_value1
    broadcasted_values = tl.broadcast_to(combined_values, [XBLOCK, RBLOCK])
    masked_values = tl.where(x_mask, broadcasted_values, float("-inf"))
    
    max_values = triton_helpers.max2(masked_values, 1)[:, None]
    shifted_values = combined_values - max_values
    exp_values = tl.math.exp(shifted_values)
    broadcasted_exp_values = tl.broadcast_to(exp_values, [XBLOCK, RBLOCK])
    masked_exp_values = tl.where(x_mask, broadcasted_exp_values, 0)
    sum_exp_values = tl.sum(masked_exp_values, 1)[:, None]
    
    tl.store(output_ptr0 + (x0 + (12608 * x1)), max_values, x_mask)
    tl.store(output_ptr1 + (x0 + (12608 * x1)), sum_exp_values, x_mask)