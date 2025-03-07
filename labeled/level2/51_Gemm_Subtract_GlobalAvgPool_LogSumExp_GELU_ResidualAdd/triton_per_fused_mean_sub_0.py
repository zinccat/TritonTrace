# From: 51_Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_mean_sub_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 512
    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[:]
    tl.full([RBLOCK], True, tl.int1)
    r1 = r_index
    x0 = x_index
    input_value = tl.load(in_ptr0 + (r1 + (512 * x0)), None)
    mean_value = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    global_avg_value = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    adjusted_value = input_value + mean_value
    result_value = adjusted_value - global_avg_value
    broadcast_result = tl.broadcast_to(result_value, [RBLOCK])
    sum_result = triton_helpers.promote_to_tensor(tl.sum(broadcast_result, 0))
    block_size = 512.0
    final_result = sum_result / block_size
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), final_result, None)