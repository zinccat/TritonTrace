# From: 46_NetVladWithGhostClusters

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_linalg_vector_norm_6per_fused_linalg_vector_norm_6(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, output_ptr0, x_num_elements, r_num_elements, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 128
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_index
    x0 = (x_index % 128)
    x1 = x_index // 128
    x3 = x_index
    temp0 = tl.load(input_ptr0 + (4 * x0 + 512 * ((r2 % 32)) + 16384 * x1 + (r2 // 32)), None, eviction_policy='evict_last')
    temp1 = tl.load(input_ptr1 + (32 * x1 + ((r2 % 32))), None, eviction_policy='evict_last')
    temp2 = tl.load(input_ptr2 + (r2 + 128 * x0), None, eviction_policy='evict_last')
    temp5 = tl.load(input_ptr3 + (32 * x1 + ((r2 % 32))), None, eviction_policy='evict_last')
    temp3 = temp1 * temp2
    temp4 = temp0 - temp3
    epsilon = 1e-12
    temp7 = triton_helpers.maximum(temp5, epsilon)
    temp8 = temp4 / temp7
    temp9 = temp8 * temp8
    temp10 = tl.broadcast_to(temp9, [XBLOCK, RBLOCK])
    temp12 = tl.sum(temp10, 1)[:, None]
    tl.store(output_ptr0 + (x3), temp12, None)