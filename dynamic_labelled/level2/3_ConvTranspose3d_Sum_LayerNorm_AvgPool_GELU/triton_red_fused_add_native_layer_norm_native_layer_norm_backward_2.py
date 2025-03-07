# From: 3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_2(
    input_grad_ptr, input_ptr, mean_ptr, variance_ptr, gamma_ptr, output_grad_ptr, output_ptr, kernel_size_0, kernel_size_1, x_num_elements, r_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = (x_index % 64)
    x1 = x_index // 64
    mean_value = tl.load(mean_ptr + (0))
    mean_broadcast = tl.broadcast_to(mean_value, [XBLOCK, RBLOCK])
    sum_grad = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = x_index
    sum_input = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r2 = r_index
        grad_input = tl.load(input_grad_ptr + (x0 + 64 * (((r2 + kernel_size_0 * kernel_size_1 * x1) % (8192 * kernel_size_0 * kernel_size_1)))), r_mask, eviction_policy='evict_first', other=0.0)
        input_value = tl.load(input_ptr + (x0 + 64 * (((r2 + kernel_size_0 * kernel_size_1 * x1) % (8192 * kernel_size_0 * kernel_size_1)))), r_mask, eviction_policy='evict_first', other=0.0)
        mean = tl.load(mean_ptr + (((r2 + kernel_size_0 * kernel_size_1 * x1) % (8192 * kernel_size_0 * kernel_size_1))), r_mask, eviction_policy='evict_last', other=0.0)
        variance = tl.load(variance_ptr + (((r2 + kernel_size_0 * kernel_size_1 * x1) % (8192 * kernel_size_0 * kernel_size_1))), r_mask, eviction_policy='evict_last', other=0.0)
        
        normalized_input = input_value + mean_broadcast
        centered_input = normalized_input - mean
        scaled_input = centered_input * variance
        grad_scaled_input = grad_input * scaled_input
        grad_scaled_input_broadcast = tl.broadcast_to(grad_scaled_input, [XBLOCK, RBLOCK])
        
        sum_grad = tl.where(r_mask, sum_grad + grad_scaled_input_broadcast, sum_grad)
        input_broadcast = tl.broadcast_to(input_value, [XBLOCK, RBLOCK])
        sum_input = tl.where(r_mask, sum_input + input_broadcast, sum_input)

    grad_output_sum = tl.sum(sum_grad, 1)[:, None]
    input_sum = tl.sum(sum_input, 1)[:, None]
    
    tl.store(output_grad_ptr + (x3), grad_output_sum, None)
    tl.store(output_ptr + (x3), input_sum, None)