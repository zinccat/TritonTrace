# From: 23_Conv3d_GroupNorm_Mean

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_0(
    input_grad_ptr, input_ptr, output_grad_ptr, output_ptr, kernel_size_0, kernel_size_1, 
    x_num_elements, r_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_div_16 = x_index // 16
    input_grad = tl.load(input_grad_ptr + (x_div_16), x_mask, eviction_policy='evict_last')
    x_full_index = x_index
    temp_sum_grad = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    temp_sum_input = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r_index_full = r_index
        input_value = tl.load(
            input_ptr + (
                (-8) * x_full_index + 
                (-2) * ((r_index_full // ((-2) + kernel_size_1)) % ((-2) + kernel_size_1)) + 
                4 * (triton_helpers.div_floor_integer(r_index_full, 4 + kernel_size_1 * kernel_size_1 + ((-4) * kernel_size_1))) + 
                kernel_size_1 * ((r_index_full // ((-2) + kernel_size_1)) % ((-2) + kernel_size_1)) + 
                kernel_size_1 * kernel_size_1 * (triton_helpers.div_floor_integer(r_index_full, 4 + kernel_size_1 * kernel_size_1 + ((-4) * kernel_size_1))) + 
                ((-4) * kernel_size_1 * (triton_helpers.div_floor_integer(r_index_full, 4 + kernel_size_1 * kernel_size_1 + ((-4) * kernel_size_1)))) + 
                ((-2) * x_full_index * kernel_size_1 * kernel_size_1) + 
                4 * kernel_size_0 * x_full_index + 
                8 * kernel_size_1 * x_full_index + 
                kernel_size_0 * x_full_index * kernel_size_1 * kernel_size_1 + 
                ((-4) * kernel_size_0 * kernel_size_1 * x_full_index) + 
                (r_index_full % ((-2) + kernel_size_1))
            ), 
            r_mask & x_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        temp_coeff = (-128) + ((-32) * kernel_size_1 * kernel_size_1) + 64 * kernel_size_0 + 128 * kernel_size_1 + ((-64) * kernel_size_0 * kernel_size_1) + 16 * kernel_size_0 * kernel_size_1 * kernel_size_1
        temp_coeff_float = temp_coeff.to(tl.float32)
        input_grad_normalized = input_grad / temp_coeff_float
        temp_product = input_grad_normalized * input_value
        temp_broadcast = tl.broadcast_to(temp_product, [XBLOCK, RBLOCK])
        temp_sum_grad_update = temp_sum_grad + temp_broadcast
        temp_sum_grad = tl.where(r_mask & x_mask, temp_sum_grad_update, temp_sum_grad)
        temp_input_broadcast = tl.broadcast_to(input_grad_normalized, [XBLOCK, RBLOCK])
        temp_sum_input_update = temp_sum_input + temp_input_broadcast
        temp_sum_input = tl.where(r_mask & x_mask, temp_sum_input_update, temp_sum_input)

    output_grad_sum = tl.sum(temp_sum_grad, 1)[:, None]
    output_input_sum = tl.sum(temp_sum_input, 1)[:, None]
    tl.store(output_grad_ptr + (x_full_index), output_grad_sum, x_mask)
    tl.store(output_ptr + (x_full_index), output_input_sum, x_mask)