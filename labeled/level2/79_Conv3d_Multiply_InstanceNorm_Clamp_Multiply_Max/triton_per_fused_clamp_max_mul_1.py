# From: 79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_clamp_max_mul_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 1612800
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 12600
    x1 = (xindex // 12600)
    x3 = xindex

    # Load input data
    input_data0 = tl.load(in_ptr0 + (x0 + (12600 * r2) + (201600 * x1)), xmask, other=0.0)
    input_data1 = tl.load(in_ptr1 + (r2), None, eviction_policy='evict_last')
    input_data2 = tl.load(in_ptr2 + (r2 + (16 * x1)), xmask, eviction_policy='evict_last', other=0.0)
    input_data3 = tl.load(in_ptr3 + (r2 + (16 * x1)), xmask, eviction_policy='evict_last', other=0.0)

    # Perform computations
    multiplied_data = input_data0 * input_data1
    subtracted_data = multiplied_data - input_data2
    multiplied_result = subtracted_data * input_data3

    # Clamp the result between -1.0 and 1.0
    clamped_min = triton_helpers.maximum(multiplied_result, -1.0)
    clamped_result = triton_helpers.minimum(clamped_min, 1.0)

    # Multiply with input_data1 and broadcast
    final_result = clamped_result * input_data1
    broadcast_result = tl.broadcast_to(final_result, [XBLOCK, RBLOCK])

    # Apply mask and find max
    masked_result = tl.where(xmask, broadcast_result, float("-inf"))
    max_result = triton_helpers.max2(masked_result, 1)[:, None]

    # Find index of max
    broadcast_rindex = tl.broadcast_to(rindex, masked_result.shape)
    _, max_index = triton_helpers.max_with_index(masked_result, broadcast_rindex, 1)
    max_index = max_index[:, None]

    # Store results
    tl.store(out_ptr0 + (x3), max_result, xmask)
    tl.store(out_ptr1 + (x0 + (12608 * x1)), max_index, xmask)