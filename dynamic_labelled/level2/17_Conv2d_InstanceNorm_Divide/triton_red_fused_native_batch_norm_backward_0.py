# From: 17_Conv2d_InstanceNorm_Divide

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_batch_norm_backward_0(
    input_grad_ptr, input_ptr, input_mean_ptr, output_grad_ptr, output_mean_ptr, kernel_size, x_num_elements, r_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = x_index
    temp_sum_grad = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_mean = tl.load(input_mean_ptr + (x0), x_mask, eviction_policy='evict_last')
    temp_sum_input_grad = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r1 = r_index
        input_grad = tl.load(input_grad_ptr + (r1 + 4*x0 + x0*kernel_size*kernel_size + ((-4)*kernel_size*x0)), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        input = tl.load(input_ptr + (r1 + 4*x0 + x0*kernel_size*kernel_size + ((-4)*kernel_size*x0)), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        scale_factor = 0.5
        scaled_input_grad = input_grad * scale_factor
        broadcast_scaled_input_grad = tl.broadcast_to(scaled_input_grad, [XBLOCK, RBLOCK])
        temp_sum_grad = temp_sum_grad + broadcast_scaled_input_grad
        temp_sum_grad = tl.where(r_mask & x_mask, temp_sum_grad, temp_sum_grad)
        
        input_mean_diff = input - input_mean
        scaled_input_mean_diff = scaled_input_grad * input_mean_diff
        broadcast_scaled_input_mean_diff = tl.broadcast_to(scaled_input_mean_diff, [XBLOCK, RBLOCK])
        temp_sum_input_grad = temp_sum_input_grad + broadcast_scaled_input_mean_diff
        temp_sum_input_grad = tl.where(r_mask & x_mask, temp_sum_input_grad, temp_sum_input_grad)

    output_grad_sum = tl.sum(temp_sum_grad, 1)[:, None]
    output_mean_sum = tl.sum(temp_sum_input_grad, 1)[:, None]
    tl.store(output_grad_ptr + (x0), output_grad_sum, x_mask)
    tl.store(output_mean_ptr + (x0), output_mean_sum, x_mask)