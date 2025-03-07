# From: 25_Conv2d_Min_Tanh_Tanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_convolution_min_tanh_tanh_backward_0(
    input_ptr0, input_ptr1, output_ptr1, output_ptr2, output_ptr3, 
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 115200
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_indices
    x0 = x_indices % 900
    x1 = (x_indices // 900)
    x3 = x_indices
    input_val0 = tl.load(input_ptr0 + (x0 + (900 * r2) + (14400 * x1)), x_mask, other=0.0)
    input_val1 = tl.load(input_ptr1 + (r2), None, eviction_policy='evict_last')
    sum_vals = input_val0 + input_val1
    broadcast_sum = tl.broadcast_to(sum_vals, [XBLOCK, RBLOCK])
    masked_broadcast = tl.where(x_mask, broadcast_sum, float("inf"))
    min_vals, _ = triton_helpers.min2(masked_broadcast, 1)[:, None]
    min_indices = tl.arange(0, RBLOCK)[None, :]
    _, min_indices_with_index = triton_helpers.min_with_index(masked_broadcast, min_indices, 1)
    min_indices_with_index = min_indices_with_index[:, None]
    tanh_min_vals = tl.extra.cuda.libdevice.tanh(min_vals)
    tanh_tanh_min_vals = tl.extra.cuda.libdevice.tanh(tanh_min_vals)
    tanh_squared = tanh_min_vals * tanh_min_vals
    one_minus_tanh_squared = 1.0 - tanh_squared
    tl.store(output_ptr2 + (x3), tanh_tanh_min_vals, x_mask)
    tl.store(output_ptr3 + (x3), one_minus_tanh_squared, x_mask)
    tl.store(output_ptr1 + (x3), min_indices_with_index, x_mask)