# From: 77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_div_mul_native_batch_norm_backward_2(
    input_grad_ptr, input_ptr, output_grad_ptr, output_ptr, 
    input_num_elements, running_mean_num_elements, 
    XBLOCK: tl.constexpr
):
    input_num_elements = 32
    running_mean_num_elements = 11
    RBLOCK: tl.constexpr = 16

    # Calculate offsets and indices
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < input_num_elements

    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < running_mean_num_elements

    # Load data with masks
    input_grad = tl.load(input_grad_ptr + (x_indices + 32 * r_indices), r_mask & x_mask, other=0.0)
    input_data = tl.load(input_ptr + (x_indices), x_mask, eviction_policy='evict_last')

    # Broadcast and apply masks
    broadcast_grad = tl.broadcast_to(input_grad, [XBLOCK, RBLOCK])
    masked_broadcast_grad = tl.where(r_mask & x_mask, broadcast_grad, 0)

    # Sum and multiply
    summed_grad = tl.sum(masked_broadcast_grad, 1)[:, None]
    multiplied_result = summed_grad * input_data

    # Store results
    tl.store(output_ptr + (x_indices), multiplied_result, x_mask)
    tl.store(output_grad_ptr + (x_indices), summed_grad, x_mask)