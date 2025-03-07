# From: 34_ConvTranspose3d_LayerNorm_GELU_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_gelu_gelu_backward_mul_native_layer_norm_native_layer_norm_backward_1(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, 
    output_ptr0, output_ptr1, kernel_size0, kernel_size1, 
    x_num_elements, r_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_mod_64 = x_index % 64
    x_div_64 = x_index // 64
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x_full_index = x_index
    temp_accumulator_gelu = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r_index_mod = r_index
        input0 = tl.load(
            input_ptr0 + (x_mod_64 + 64 * (((r_index_mod + kernel_size0 * kernel_size1 * x_div_64) % (8192 * kernel_size0 * kernel_size1)))), 
            r_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        input1 = tl.load(
            input_ptr1 + (x_mod_64 + 64 * (((r_index_mod + kernel_size0 * kernel_size1 * x_div_64) % (8192 * kernel_size0 * kernel_size1)))), 
            r_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        input2 = tl.load(
            input_ptr2 + (x_mod_64 + 64 * (((r_index_mod + kernel_size0 * kernel_size1 * x_div_64) % (8192 * kernel_size0 * kernel_size1)))), 
            r_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        input3 = tl.load(
            input_ptr3 + (((r_index_mod + kernel_size0 * kernel_size1 * x_div_64) % (8192 * kernel_size0 * kernel_size1))), 
            r_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        input4 = tl.load(
            input_ptr4 + (((r_index_mod + kernel_size0 * kernel_size1 * x_div_64) % (8192 * kernel_size0 * kernel_size1))), 
            r_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        scale_factor = 1.0
        scaled_input0 = input0 * scale_factor
        gelu_coeff = 0.7071067811865476
        scaled_input1 = input1 * gelu_coeff
        erf_result = tl.extra.cuda.libdevice.erf(scaled_input1)
        erf_plus_one = erf_result + 1.0
        half = 0.5
        gelu_approx = erf_plus_one * half
        input1_squared = input1 * input1
        neg_half = -0.5
        exp_component = tl.math.exp(input1_squared * neg_half)
        sqrt_two_pi = 0.3989422804014327
        gaussian_component = exp_component * sqrt_two_pi
        input1_gaussian = input1 * gaussian_component
        gelu_final = gelu_approx + input1_gaussian
        gelu_scaled = scaled_input0 * gelu_final
        input2_diff = input2 - input3
        input2_diff_scaled = input2_diff * input4
        gelu_diff_scaled = gelu_scaled * input2_diff_scaled
        broadcast_gelu_diff_scaled = tl.broadcast_to(gelu_diff_scaled, [XBLOCK, RBLOCK])
        temp_accumulator = tl.where(r_mask, temp_accumulator + broadcast_gelu_diff_scaled, temp_accumulator)
        broadcast_gelu_scaled = tl.broadcast_to(gelu_scaled, [XBLOCK, RBLOCK])
        temp_accumulator_gelu = tl.where(r_mask, temp_accumulator_gelu + broadcast_gelu_scaled, temp_accumulator_gelu)

    sum_temp_accumulator = tl.sum(temp_accumulator, 1)[:, None]
    sum_temp_accumulator_gelu = tl.sum(temp_accumulator_gelu, 1)[:, None]
    tl.store(output_ptr0 + (x_full_index), sum_temp_accumulator, None)
    tl.store(output_ptr1 + (x_full_index), sum_temp_accumulator_gelu, None)