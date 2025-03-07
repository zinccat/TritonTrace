# From: 51_Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_sum_0(input_ptr, output_ptr, num_elements, residual_num_elements):
    BLOCK_SIZE_X: tl.constexpr = 1
    BLOCK_SIZE_R: tl.constexpr = 1024
    x_offset = tl.program_id(0) * BLOCK_SIZE_X
    x_index = tl.full([1], x_offset, tl.int32)
    tl.full([BLOCK_SIZE_R], True, tl.int1)
    r_index = tl.arange(0, BLOCK_SIZE_R)[:]
    tl.full([BLOCK_SIZE_R], True, tl.int1)
    residual_index = r_index
    x_base_index = x_index
    temp_value = tl.load(input_ptr + (residual_index + 1024 * x_base_index), None)
    broadcasted_temp = tl.broadcast_to(temp_value, [BLOCK_SIZE_R])
    reduced_sum = triton_helpers.promote_to_tensor(tl.sum(broadcasted_temp, 0))
    tl.store(output_ptr + (x_base_index), reduced_sum, None)