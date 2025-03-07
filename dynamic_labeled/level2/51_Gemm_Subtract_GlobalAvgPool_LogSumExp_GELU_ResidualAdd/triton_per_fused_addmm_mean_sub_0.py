# From: 51_Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_addmm_mean_sub_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 512
    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[:]
    tl.full([RBLOCK], True, tl.int1)
    r1 = r_index
    x0 = x_index
    input_tensor0 = tl.load(in_ptr0 + (r1 + 512 * x0), None)
    input_tensor1 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    input_tensor2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    sum_tensors = input_tensor0 + input_tensor1
    subtracted_tensor = sum_tensors - input_tensor2
    broadcasted_tensor = tl.broadcast_to(subtracted_tensor, [RBLOCK])
    sum_broadcasted = triton_helpers.promote_to_tensor(tl.sum(broadcasted_tensor, 0))
    block_size = 512.0
    mean_result = sum_broadcasted / block_size
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), mean_result, None)