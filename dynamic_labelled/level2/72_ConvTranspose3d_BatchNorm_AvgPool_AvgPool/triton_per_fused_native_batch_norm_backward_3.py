# From: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_batch_norm_backward_3(
    input_grad_ptr, input_ptr, output_grad_ptr, output_ptr, 
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 16
    rnumel = 249
    RBLOCK: tl.constexpr = 256

    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel

    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < rnumel

    r_index = r_indices
    x_index = x_indices

    tmp_input_grad = tl.load(input_grad_ptr + (r_index + 249 * x_index), r_mask & x_mask, other=0.0)
    tmp_input = tl.load(input_ptr + (x_index), x_mask, eviction_policy='evict_last')

    tmp_broadcast = tl.broadcast_to(tmp_input_grad, [XBLOCK, RBLOCK])
    tmp_masked = tl.where(r_mask & x_mask, tmp_broadcast, 0)
    tmp_sum = tl.sum(tmp_masked, 1)[:, None]

    tmp_output = tmp_sum * tmp_input
    tl.store(output_ptr + (x_index), tmp_output, x_mask)
    tl.store(output_grad_ptr + (x_index), tmp_sum, x_mask)