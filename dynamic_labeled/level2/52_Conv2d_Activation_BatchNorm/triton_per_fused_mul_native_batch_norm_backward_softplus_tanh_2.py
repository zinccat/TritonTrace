# From: 52_Conv2d_Activation_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_mul_native_batch_norm_backward_softplus_tanh_2(
    input_grad_ptr, input_ptr, output_grad_ptr, output_ptr, 
    input_num_elements, running_num_elements, XBLOCK: tl.constexpr
):
    input_num_elements = 16
    running_num_elements = 15
    RBLOCK: tl.constexpr = 16

    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < input_num_elements

    running_indices = tl.arange(0, RBLOCK)[None, :]
    running_mask = running_indices < running_num_elements

    running_index = running_indices
    input_index = input_indices

    input_grad = tl.load(input_grad_ptr + (input_index + 16 * running_index), running_mask & input_mask, other=0.0)
    input_data = tl.load(input_ptr + (input_index), input_mask, eviction_policy='evict_last')

    broadcast_grad = tl.broadcast_to(input_grad, [XBLOCK, RBLOCK])
    masked_broadcast_grad = tl.where(running_mask & input_mask, broadcast_grad, 0)
    summed_grad = tl.sum(masked_broadcast_grad, 1)[:, None]

    output_grad = summed_grad * input_data
    tl.store(output_ptr + (input_index), output_grad, input_mask)

    tl.store(output_grad_ptr + (input_index), summed_grad, input_mask)