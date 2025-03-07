# From: 18_SqueezeNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_mean_relu_28red_fused_convolution_mean_relu_28(
    input_ptr0, input_ptr1, output_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 2000
    rnumel = 85
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x1_indices = x_indices // 1000
    x0_indices = x_indices % 1000
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3_indices = x_indices

    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r2_indices = r_indices
        temp_index0 = r2_indices + 85 * x1_indices
        temp_limit = tl.full([1, 1], 169, tl.int32)
        temp_mask = temp_index0 < temp_limit
        temp_load0 = tl.load(
            input_ptr0 + (x0_indices + 1000 * ((temp_index0 % 169))),
            r_mask & temp_mask & x_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        temp_load1 = tl.load(
            input_ptr1 + (tl.broadcast_to(x0_indices, [XBLOCK, RBLOCK])),
            r_mask & temp_mask & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        temp_summed = temp_load0 + temp_load1
        temp_zero = tl.full([1, 1], 0, tl.int32)
        temp_max = triton_helpers.maximum(temp_zero, temp_summed)
        temp_broadcast = tl.full(temp_max.shape, 0, temp_max.dtype)
        temp_result = tl.where(temp_mask, temp_max, temp_broadcast)
        temp_broadcasted = tl.broadcast_to(temp_result, [XBLOCK, RBLOCK])
        temp_accumulated = temp_sum + temp_broadcasted
        temp_sum = tl.where(r_mask & x_mask, temp_accumulated, temp_sum)

    temp_final_sum = tl.sum(temp_sum, 1)[:, None]
    tl.store(output_ptr0 + (x3_indices), temp_final_sum, x_mask)