# From: 84_Gemm_BatchNorm_Scaling_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__softmax_backward_data_mul_native_batch_norm_backward_2red_fused__softmax_backward_data_mul_native_batch_norm_backward_2(
    input_grad_ptr, input_data_ptr, scale_ptr, bias_ptr, running_mean_ptr, running_var_ptr, input_ptr, 
    output_grad_ptr, output_data_ptr, output_ptr, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):

    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x_indices = xindex
    broadcast_bias = tl.load(bias_ptr + (0))
    broadcast_bias_expanded = tl.broadcast_to(broadcast_bias, [XBLOCK, RBLOCK])
    temp_grad_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    running_var = tl.load(running_var_ptr + (x_indices), xmask, eviction_policy='evict_last')
    temp_data_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r_indices = rindex
        grad_input = tl.load(input_grad_ptr + (x_indices + 512 * r_indices), rmask & xmask, eviction_policy='evict_first', other=0.0)
        scale_value = tl.load(scale_ptr + (r_indices), rmask, eviction_policy='evict_last', other=0.0)
        input_data = tl.load(input_data_ptr + (x_indices + 512 * r_indices), rmask & xmask, eviction_policy='evict_first', other=0.0)
        running_mean_value = tl.load(running_mean_ptr + (x_indices + 512 * r_indices), rmask & xmask, eviction_policy='evict_first', other=0.0)
        
        neg_grad_input = -grad_input
        input_grad_scaled = input_data * grad_input
        scaled_grad_input = tl.extra.cuda.libdevice.fma(neg_grad_input, scale_value, input_grad_scaled)
        scaled_grad_input_expanded = scaled_grad_input * broadcast_bias_expanded
        broadcast_scaled_grad_input = tl.broadcast_to(scaled_grad_input_expanded, [XBLOCK, RBLOCK])
        temp_grad_accumulator_updated = temp_grad_accumulator + broadcast_scaled_grad_input
        temp_grad_accumulator = tl.where(rmask & xmask, temp_grad_accumulator_updated, temp_grad_accumulator)
        
        running_mean_diff = running_mean_value - running_var
        running_mean_diff_scaled = scaled_grad_input_expanded * running_mean_diff
        broadcast_running_mean_diff_scaled = tl.broadcast_to(running_mean_diff_scaled, [XBLOCK, RBLOCK])
        temp_data_accumulator_updated = temp_data_accumulator + broadcast_running_mean_diff_scaled
        temp_data_accumulator = tl.where(rmask & xmask, temp_data_accumulator_updated, temp_data_accumulator)

    grad_output_sum = tl.sum(temp_grad_accumulator, 1)[:, None]
    data_output_sum = tl.sum(temp_data_accumulator, 1)[:, None]
    tl.store(output_grad_ptr + (x_indices), grad_output_sum, xmask)
    tl.store(output_data_ptr + (x_indices), data_output_sum, xmask)
    
    input_ptr_value = tl.load(input_ptr + (x_indices), xmask, eviction_policy='evict_last')
    final_output = data_output_sum * input_ptr_value
    tl.store(output_ptr + (x_indices), final_output, xmask)