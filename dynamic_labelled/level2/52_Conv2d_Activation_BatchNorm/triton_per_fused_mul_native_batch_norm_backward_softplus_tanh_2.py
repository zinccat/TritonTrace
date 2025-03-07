# From: 52_Conv2d_Activation_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_mul_native_batch_norm_backward_softplus_tanh_2(
    input_grad_ptr, input_ptr, output_grad_ptr, output_ptr, 
    input_num_elements, running_mean_num_elements, 
    XBLOCK: tl.constexpr
):
    input_num_elements = 16
    running_mean_num_elements = 15
    RBLOCK: tl.constexpr = 16

    # Calculate offsets and indices
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < input_num_elements

    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < running_mean_num_elements

    # Load data
    input_grad = tl.load(input_grad_ptr + (x_indices + 16 * r_indices), r_mask & x_mask, other=0.0)
    input_data = tl.load(input_ptr + (x_indices), x_mask, eviction_policy='evict_last')

    # Broadcast and compute
    broadcast_input_grad = tl.broadcast_to(input_grad, [XBLOCK, RBLOCK])
    masked_broadcast = tl.where(r_mask & x_mask, broadcast_input_grad, 0)
    sum_over_r = tl.sum(masked_broadcast, 1)[:, None]

    # Compute output gradients
    output_grad = sum_over_r * input_data
    tl.store(output_ptr + (x_indices), output_grad, x_mask)
    tl.store(output_grad_ptr + (x_indices), sum_over_r, x_mask)