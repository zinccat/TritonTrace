# From: 73_Conv2d_BatchNorm_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_mul_native_batch_norm_backward_2(input_grad_ptr, input_ptr, output_grad_ptr, output_ptr, num_elements, reduced_num_elements, XBLOCK: tl.constexpr):
    num_elements = 16
    reduced_num_elements = 15
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < reduced_num_elements
    reduced_indices = r_indices
    input_indices = x_indices
    input_grad = tl.load(input_grad_ptr + (input_indices + 16 * reduced_indices), r_mask & x_mask, other=0.0)
    input_data = tl.load(input_ptr + (input_indices), x_mask, eviction_policy='evict_last')
    broadcasted_input_grad = tl.broadcast_to(input_grad, [XBLOCK, RBLOCK])
    masked_broadcasted_input_grad = tl.where(r_mask & x_mask, broadcasted_input_grad, 0)
    summed_masked_broadcasted_input_grad = tl.sum(masked_broadcasted_input_grad, 1)[:, None]
    output_grad = summed_masked_broadcasted_input_grad * input_data
    tl.store(output_ptr + (input_indices), output_grad, x_mask)
    tl.store(output_grad_ptr + (input_indices), summed_masked_broadcasted_input_grad, x_mask)