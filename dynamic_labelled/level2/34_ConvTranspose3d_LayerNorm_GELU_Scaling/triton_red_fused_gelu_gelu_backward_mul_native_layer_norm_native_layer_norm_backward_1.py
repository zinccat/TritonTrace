# From: 34_ConvTranspose3d_LayerNorm_GELU_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_gelu_gelu_backward_mul_native_layer_norm_native_layer_norm_backward_1(
    input_grad0_ptr, input_grad1_ptr, input_grad2_ptr, input_grad3_ptr, input_grad4_ptr,
    output_grad0_ptr, output_grad1_ptr, kernel_size0, kernel_size1, x_num_elements, r_num_elements,
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_mod_64 = x_index % 64
    x_div_64 = x_index // 64
    temp_accumulator0 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x_full_index = x_index
    temp_accumulator1 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r_index_mod = r_index
        input_grad0 = tl.load(
            input_grad0_ptr + (x_mod_64 + 64 * (((r_index_mod + kernel_size0 * kernel_size1 * x_div_64) % (8192 * kernel_size0 * kernel_size1)))),
            r_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        input_grad1 = tl.load(
            input_grad1_ptr + (x_mod_64 + 64 * (((r_index_mod + kernel_size0 * kernel_size1 * x_div_64) % (8192 * kernel_size0 * kernel_size1)))),
            r_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        input_grad2 = tl.load(
            input_grad2_ptr + (x_mod_64 + 64 * (((r_index_mod + kernel_size0 * kernel_size1 * x_div_64) % (8192 * kernel_size0 * kernel_size1)))),
            r_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        input_grad3 = tl.load(
            input_grad3_ptr + (((r_index_mod + kernel_size0 * kernel_size1 * x_div_64) % (8192 * kernel_size0 * kernel_size1))),
            r_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        input_grad4 = tl.load(
            input_grad4_ptr + (((r_index_mod + kernel_size0 * kernel_size1 * x_div_64) % (8192 * kernel_size0 * kernel_size1))),
            r_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        scale_factor = 1.0
        scaled_input_grad0 = input_grad0 * scale_factor
        gelu_coeff = 0.7071067811865476
        scaled_input_grad1 = input_grad1 * gelu_coeff
        erf_result = tl.extra.cuda.libdevice.erf(scaled_input_grad1)
        erf_plus_one = erf_result + 1.0
        half = 0.5
        half_erf_plus_one = erf_plus_one * half
        input_grad1_squared = input_grad1 * input_grad1
        neg_half = -0.5
        exp_component = input_grad1_squared * neg_half
        exp_result = tl.math.exp(exp_component)
        sqrt_two_pi = 0.3989422804014327
        gaussian_component = exp_result * sqrt_two_pi
        scaled_gaussian_component = input_grad1 * gaussian_component
        gelu_result = half_erf_plus_one + scaled_gaussian_component
        elementwise_product = scaled_input_grad0 * gelu_result
        input_grad2_diff = input_grad2 - input_grad3
        elementwise_product_with_grad4 = input_grad2_diff * input_grad4
        final_product = elementwise_product * elementwise_product_with_grad4
        broadcasted_final_product = tl.broadcast_to(final_product, [XBLOCK, RBLOCK])
        temp_accumulator0 = temp_accumulator0 + broadcasted_final_product
        broadcasted_elementwise_product = tl.broadcast_to(elementwise_product, [XBLOCK, RBLOCK])
        temp_accumulator1 = temp_accumulator1 + broadcasted_elementwise_product

        temp_accumulator0 = tl.where(r_mask, temp_accumulator0, _tmp26)
        temp_accumulator1 = tl.where(r_mask, temp_accumulator1, _tmp29)

    summed_temp_accumulator0 = tl.sum(temp_accumulator0, 1)[:, None]
    summed_temp_accumulator1 = tl.sum(temp_accumulator1, 1)[:, None]
    tl.store(output_grad0_ptr + (x_full_index), summed_temp_accumulator0, None)
    tl.store(output_grad1_ptr + (x_full_index), summed_temp_accumulator1, None)