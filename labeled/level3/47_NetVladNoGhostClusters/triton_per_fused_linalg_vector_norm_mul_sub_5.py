# From: 47_NetVladNoGhostClusters

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_linalg_vector_norm_mul_sub_5per_fused_linalg_vector_norm_mul_sub_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 512
    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    r_mask = tl.full([RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[:]
    r_mask = tl.full([RBLOCK], True, tl.int1)
    r2 = r_index
    x3 = x_index
    x0 = (x_index % 32)
    input_val0 = tl.load(in_ptr0 + (r2 + 512 * x3), None)
    input_val1 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    input_val2 = tl.load(in_ptr2 + (x0 + 32 * r2), None, eviction_policy='evict_last')
    product = input_val1 * input_val2
    difference = input_val0 - product
    squared_difference = difference * difference
    broadcasted_squares = tl.broadcast_to(squared_difference, [RBLOCK])
    sum_of_squares = triton_helpers.promote_to_tensor(tl.sum(broadcasted_squares, 0))
    sqrt_result = tl.extra.cuda.libdevice.sqrt(sum_of_squares)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), sqrt_result, None)