# From: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_batch_norm_backward_tanh_backward_7(
    input_grad_ptr, input_ptr, output_grad_ptr, output_ptr, 
    input_num_elements, running_mean_num_elements, 
    XBLOCK: tl.constexpr
):
    input_num_elements = 64
    running_mean_num_elements = 6
    RBLOCK: tl.constexpr = 8

    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < input_num_elements

    running_mean_indices = tl.arange(0, RBLOCK)[None, :]
    running_mean_mask = running_mean_indices < running_mean_num_elements

    running_mean_index = running_mean_indices
    input_index = input_indices

    input_grad = tl.load(input_grad_ptr + (input_index + 64 * running_mean_index), 
                         running_mean_mask & input_mask, other=0.0)
    input_data = tl.load(input_ptr + (input_index), input_mask, eviction_policy='evict_last')

    broadcast_input_grad = tl.broadcast_to(input_grad, [XBLOCK, RBLOCK])
    masked_broadcast_input_grad = tl.where(running_mean_mask & input_mask, broadcast_input_grad, 0)
    summed_grad = tl.sum(masked_broadcast_input_grad, 1)[:, None]

    output_grad = summed_grad * input_data
    tl.store(output_ptr + (input_index), output_grad, input_mask)

    tl.store(output_grad_ptr + (input_index), summed_grad, input_mask)