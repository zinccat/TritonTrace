# From: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_batch_norm_backward_3(input_grad_ptr, input_ptr, output_grad_ptr, output_ptr, num_elements, reduced_num_elements, XBLOCK: tl.constexpr):
    num_elements = 16
    reduced_num_elements = 249
    RBLOCK: tl.constexpr = 256

    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements

    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < reduced_num_elements

    reduced_indices = r_indices
    input_indices = x_indices

    input_grad = tl.load(input_grad_ptr + (reduced_indices + 249 * input_indices), r_mask & x_mask, other=0.0)
    input_data = tl.load(input_ptr + (input_indices), x_mask, eviction_policy='evict_last')

    broadcast_input_grad = tl.broadcast_to(input_grad, [XBLOCK, RBLOCK])
    masked_broadcast = tl.where(r_mask & x_mask, broadcast_input_grad, 0)
    summed_masked_broadcast = tl.sum(masked_broadcast, 1)[:, None]

    output_data = summed_masked_broadcast * input_data

    tl.store(output_ptr + (input_indices), output_data, x_mask)
    tl.store(output_grad_ptr + (input_indices), summed_masked_broadcast, x_mask)