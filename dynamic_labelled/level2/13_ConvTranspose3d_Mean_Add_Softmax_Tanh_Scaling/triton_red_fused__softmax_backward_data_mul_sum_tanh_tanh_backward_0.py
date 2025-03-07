# From: 13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__softmax_backward_data_mul_sum_tanh_tanh_backward_0(
    input_grad_ptr, input_ptr, output_ptr, kernel_size_0, kernel_size_1, x_num_elements, r_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_num_elements = 241
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < x_num_elements
    r_base_indices = tl.arange(0, RBLOCK)[None, :]
    x0_indices = x_indices
    _accumulated_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, r_num_elements, RBLOCK):
        r_indices = r_offset + r_base_indices
        r_mask = r_indices < r_num_elements
        r1_indices = r_indices
        index_offset = r1_indices + x0_indices * (
            triton_helpers.div_floor_integer(
                240 + ((-1) * kernel_size_0) + 2 * kernel_size_0 * kernel_size_0 + 
                ((-8) * kernel_size_1 * kernel_size_0 * kernel_size_0) + 
                ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                4 * kernel_size_0 * kernel_size_1 + 
                8 * kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1, 
                241
            )
        )
        max_index = ((-1) * kernel_size_0) + 2 * kernel_size_0 * kernel_size_0 + 
                    ((-8) * kernel_size_1 * kernel_size_0 * kernel_size_0) + 
                    ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                    4 * kernel_size_0 * kernel_size_1 + 
                    8 * kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1
        index_mask = index_offset < max_index

        input_grad_value = tl.load(
            input_grad_ptr + (
                (((-1) * (((index_offset // ((-1) + 2 * kernel_size_1)) % ((-1) + 2 * kernel_size_1)))) + 
                 ((-1) * (((index_offset // ((-1) + ((-4) * kernel_size_1 * kernel_size_1) + 2 * kernel_size_0 + 4 * kernel_size_1 + ((-8) * kernel_size_0 * kernel_size_1) + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1)) % kernel_size_0))) + 
                 ((-4) * kernel_size_1 * (((index_offset // (1 + ((-4) * kernel_size_1) + 4 * kernel_size_1 * kernel_size_1)) % ((-1) + 2 * kernel_size_0)))) + 
                 ((-4) * kernel_size_1 * kernel_size_1 * (((index_offset // ((-1) + ((-4) * kernel_size_1 * kernel_size_1) + 2 * kernel_size_0 + 4 * kernel_size_1 + ((-8) * kernel_size_0 * kernel_size_1) + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1)) % kernel_size_0))) + 
                 2 * kernel_size_0 * (((index_offset // ((-1) + ((-4) * kernel_size_1 * kernel_size_1) + 2 * kernel_size_0 + 4 * kernel_size_1 + ((-8) * kernel_size_0 * kernel_size_1) + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1)) % kernel_size_0)) + 
                 2 * kernel_size_1 * (((index_offset // ((-1) + 2 * kernel_size_1)) % ((-1) + 2 * kernel_size_1))) + 
                 4 * kernel_size_1 * (((index_offset // ((-1) + ((-4) * kernel_size_1 * kernel_size_1) + 2 * kernel_size_0 + 4 * kernel_size_1 + ((-8) * kernel_size_0 * kernel_size_1) + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1)) % kernel_size_0)) + 
                 4 * kernel_size_1 * kernel_size_1 * (((index_offset // (1 + ((-4) * kernel_size_1) + 4 * kernel_size_1 * kernel_size_1)) % ((-1) + 2 * kernel_size_0))) + 
                 ((-8) * kernel_size_0 * kernel_size_1 * (((index_offset // ((-1) + ((-4) * kernel_size_1 * kernel_size_1) + 2 * kernel_size_0 + 4 * kernel_size_1 + ((-8) * kernel_size_0 * kernel_size_1) + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1)) % kernel_size_0))) + 
                 8 * kernel_size_0 * kernel_size_1 * kernel_size_1 * (((index_offset // ((-1) + ((-4) * kernel_size_1 * kernel_size_1) + 2 * kernel_size_0 + 4 * kernel_size_1 + ((-8) * kernel_size_0 * kernel_size_1) + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1)) % kernel_size_0)) + 
                 ((index_offset % ((-1) + 2 * kernel_size_1))) + 
                 (((index_offset // (1 + ((-4) * kernel_size_1) + 4 * kernel_size_1 * kernel_size_1)) % ((-1) + 2 * kernel_size_0))))
            ),
            rmask & index_mask & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        neg_input_grad_value = -input_grad_value

        input_value = tl.load(
            input_ptr + (
                (((-1) * (((index_offset // ((-1) + 2 * kernel_size_1)) % ((-1) + 2 * kernel_size_1)))) + 
                 ((-1) * (((index_offset // ((-1) + ((-4) * kernel_size_1 * kernel_size_1) + 2 * kernel_size_0 + 4 * kernel_size_1 + ((-8) * kernel_size_0 * kernel_size_1) + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1)) % kernel_size_0))) + 
                 ((-4) * kernel_size_1 * (((index_offset // (1 + ((-4) * kernel_size_1) + 4 * kernel_size_1 * kernel_size_1)) % ((-1) + 2 * kernel_size_0)))) + 
                 ((-4) * kernel_size_1 * kernel_size_1 * (((index_offset // ((-1) + ((-4) * kernel_size_1 * kernel_size_1) + 2 * kernel_size_0 + 4 * kernel_size_1 + ((-8) * kernel_size_0 * kernel_size_1) + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1)) % kernel_size_0))) + 
                 2 * kernel_size_0 * (((index_offset // ((-1) + ((-4) * kernel_size_1 * kernel_size_1) + 2 * kernel_size_0 + 4 * kernel_size_1 + ((-8) * kernel_size_0 * kernel_size_1) + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1)) % kernel_size_0)) + 
                 2 * kernel_size_1 * (((index_offset // ((-1) + 2 * kernel_size_1)) % ((-1) + 2 * kernel_size_1))) + 
                 4 * kernel_size_1 * (((index_offset // ((-1) + ((-4) * kernel_size_1 * kernel_size_1) + 2 * kernel_size_0 + 4 * kernel_size_1 + ((-8) * kernel_size_0 * kernel_size_1) + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1)) % kernel_size_0)) + 
                 4 * kernel_size_1 * kernel_size_1 * (((index_offset // (1 + ((-4) * kernel_size_1) + 4 * kernel_size_1 * kernel_size_1)) % ((-1) + 2 * kernel_size_0))) + 
                 ((-8) * kernel_size_0 * kernel_size_1 * (((index_offset // ((-1) + ((-4) * kernel_size_1 * kernel_size_1) + 2 * kernel_size_0 + 4 * kernel_size_1 + ((-8) * kernel_size_0 * kernel_size_1) + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1)) % kernel_size_0))) + 
                 8 * kernel_size_0 * kernel_size_1 * kernel_size_1 * (((index_offset // ((-1) + ((-4) * kernel_size_1 * kernel_size_1) + 2 * kernel_size_0 + 4 * kernel_size_1 + ((-8) * kernel_size_0 * kernel_size_1) + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1)) % kernel_size_0)) + 
                 ((index_offset % ((-1) + 2 * kernel_size_1))) + 
                 (((index_offset // (1 + ((-4) * kernel_size_1) + 4 * kernel_size_1 * kernel_size_1)) % ((-1) + 2 * kernel_size_0))))
            ),
            rmask & index_mask & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        scale_factor = 2.0
        scaled_input_value = input_value * scale_factor

        tanh_input_grad_value = tl.extra.cuda.libdevice.tanh(input_grad_value)
        tanh_squared = tanh_input_grad_value * tanh_input_grad_value
        one_minus_tanh_squared = 1.0 - tanh_squared

        scaled_tanh_term = scaled_input_value * one_minus_tanh_squared
        tanh_term = scaled_tanh_term * input_grad_value

        fused_tanh_term = tl.extra.cuda.libdevice.fma(neg_input_grad_value, tanh_term, tanh_term)

        zero_filled_fused_tanh = tl.full(fused_tanh_term.shape, 0, fused_tanh_term.dtype)
        selected_fused_tanh = tl.where(index_mask, fused_tanh_term, zero_filled_fused_tanh)

        broadcasted_fused_tanh = tl.broadcast_to(selected_fused_tanh, [XBLOCK, RBLOCK])
        accumulated_sum = _accumulated_sum + broadcasted_fused_tanh

        _accumulated_sum = tl.where(rmask & x_mask, accumulated_sum, _accumulated_sum)

    summed_accumulated = tl.sum(_accumulated_sum, 1)[:, None]
    tl.store(output_ptr + (x0_indices), summed_accumulated, x_mask)