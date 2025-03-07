# From: 55_Matmul_MaxPool_Sum_Scale

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel

    col_index = xindex % 2
    row_index = xindex // 2
    linear_index = xindex

    input_value_0 = tl.load(in_ptr0 + (2 * col_index + 5 * row_index), xmask, eviction_policy='evict_last')
    input_value_1 = tl.load(in_ptr0 + (1 + 2 * col_index + 5 * row_index), xmask, eviction_policy='evict_last')

    is_greater = input_value_1 > input_value_0
    true_value = tl.full([1], 1, tl.int8)
    false_value = tl.full([1], 0, tl.int8)

    max_pool_result = tl.where(is_greater, true_value, false_value)
    triton_helpers.maximum(input_value_1, input_value_0)

    tl.store(out_ptr0 + (linear_index), max_pool_result, xmask)