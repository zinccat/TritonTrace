# From: 45_UNetSoftmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_22poi_fused_cat_22(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    x1 = ((x_index // 2048) % 512)
    x0 = (x_index % 2048)
    x2 = x_index // 1048576
    x3 = x_index
    
    tmp0 = x1
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 256, tl.int64)
    condition1 = tmp0 < tmp3
    
    value0 = tl.load(in_ptr0 + (x0 + 2048 * x1 + 524288 * x2), condition1, other=0.0)
    value1 = tl.load(in_ptr1 + (x1), condition1, eviction_policy='evict_last', other=0.0)
    sum_values = value0 + value1
    
    zero_filled = tl.full(sum_values.shape, 0.0, sum_values.dtype)
    selected_value = tl.where(condition1, sum_values, zero_filled)
    
    condition2 = tmp0 >= tmp3
    tmp13 = tl.load(in_ptr2 + (x0 + 2048 * ((-256) + x1) + 524288 * x2), condition2, other=0.0)
    final_value = tl.where(condition1, selected_value, tmp13)
    
    tl.store(out_ptr0 + (x3), final_value, None)