# From: 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_mul_sum_1(input_ptr, output_ptr, total_elements, reduced_elements):
    BLOCK_SIZE_X: tl.constexpr = 1
    reduced_elements = 310
    BLOCK_SIZE_R: tl.constexpr = 512
    x_offset = tl.program_id(0) * BLOCK_SIZE_X
    tl.full([1], x_offset, tl.int32)
    tl.full([BLOCK_SIZE_R], True, tl.int1)
    r_index = tl.arange(0, BLOCK_SIZE_R)[:]
    r_mask = r_index < reduced_elements
    r0 = r_index
    loaded_values = tl.load(input_ptr + (r0), r_mask, other=0.0)
    broadcasted_values = tl.broadcast_to(loaded_values, [BLOCK_SIZE_R])
    masked_values = tl.where(r_mask, broadcasted_values, 0)
    summed_result = triton_helpers.promote_to_tensor(tl.sum(masked_values, 0))
    tl.store(output_ptr + (tl.full([1], 0, tl.int32)), summed_result, None)