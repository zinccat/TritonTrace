# From: 83_Conv3d_GroupNorm_Min_Clamp_Dropout

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_1(
    input_grad_ptr, input_ptr, mask_ptr, weight_ptr, output_grad_ptr0, output_grad_ptr1, kernel_size0, kernel_size1, x_num_elements, r_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = x_index
    temp_grad_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    temp_weight_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r1 = r_index

        input_grad = tl.load(
            input_grad_ptr + (((-8) * x0) + ((-2) * (((r1 // ((-2) + kernel_size1)) % ((-2) + kernel_size1)))) +
                              4 * (triton_helpers.div_floor_integer(r1, 4 + kernel_size1 * kernel_size1 + ((-4) * kernel_size1))) +
                              kernel_size1 * (((r1 // ((-2) + kernel_size1)) % ((-2) + kernel_size1))) +
                              kernel_size1 * kernel_size1 * (triton_helpers.div_floor_integer(r1, 4 + kernel_size1 * kernel_size1 + ((-4) * kernel_size1))) +
                              ((-4) * kernel_size1 * (triton_helpers.div_floor_integer(r1, 4 + kernel_size1 * kernel_size1 + ((-4) * kernel_size1)))) +
                              ((-2) * x0 * kernel_size1 * kernel_size1) + 4 * kernel_size0 * x0 + 8 * kernel_size1 * x0 +
                              kernel_size0 * x0 * kernel_size1 * kernel_size1 + ((-4) * kernel_size0 * kernel_size1 * x0) +
                              ((r1 % ((-2) + kernel_size1)))),
            r_mask & x_mask, eviction_policy='evict_last', other=0.0
        )

        input_value = tl.load(
            input_ptr + (((-8) * x0) + ((-2) * (((r1 // ((-2) + kernel_size1)) % ((-2) + kernel_size1)))) +
                         4 * (triton_helpers.div_floor_integer(r1, 4 + kernel_size1 * kernel_size1 + ((-4) * kernel_size1))) +
                         kernel_size1 * (((r1 // ((-2) + kernel_size1)) % ((-2) + kernel_size1))) +
                         kernel_size1 * kernel_size1 * (triton_helpers.div_floor_integer(r1, 4 + kernel_size1 * kernel_size1 + ((-4) * kernel_size1))) +
                         ((-4) * kernel_size1 * (triton_helpers.div_floor_integer(r1, 4 + kernel_size1 * kernel_size1 + ((-4) * kernel_size1)))) +
                         ((-2) * x0 * kernel_size1 * kernel_size1) + 4 * kernel_size0 * x0 + 8 * kernel_size1 * x0 +
                         kernel_size0 * x0 * kernel_size1 * kernel_size1 + ((-4) * kernel_size0 * kernel_size1 * x0) +
                         ((r1 % ((-2) + kernel_size1)))),
            r_mask & x_mask, eviction_policy='evict_last', other=0.0
        )

        mask_value = tl.load(
            mask_ptr + (((-8) * x0) + ((-2) * (((r1 // ((-2) + kernel_size1)) % ((-2) + kernel_size1)))) +
                        4 * (triton_helpers.div_floor_integer(r1, 4 + kernel_size1 * kernel_size1 + ((-4) * kernel_size1))) +
                        kernel_size1 * (((r1 // ((-2) + kernel_size1)) % ((-2) + kernel_size1))) +
                        kernel_size1 * kernel_size1 * (triton_helpers.div_floor_integer(r1, 4 + kernel_size1 * kernel_size1 + ((-4) * kernel_size1))) +
                        ((-4) * kernel_size1 * (triton_helpers.div_floor_integer(r1, 4 + kernel_size1 * kernel_size1 + ((-4) * kernel_size1)))) +
                        ((-2) * x0 * kernel_size1 * kernel_size1) + 4 * kernel_size0 * x0 + 8 * kernel_size1 * x0 +
                        kernel_size0 * x0 * kernel_size1 * kernel_size1 + ((-4) * kernel_size0 * kernel_size1 * x0) +
                        ((r1 % ((-2) + kernel_size1)))),
            r_mask & x_mask, eviction_policy='evict_last', other=0.0
        ).to(tl.int1)

        weight_value = tl.load(
            weight_ptr + (((-8) * x0) + ((-2) * (((r1 // ((-2) + kernel_size1)) % ((-2) + kernel_size1)))) +
                          4 * (triton_helpers.div_floor_integer(r1, 4 + kernel_size1 * kernel_size1 + ((-4) * kernel_size1))) +
                          kernel_size1 * (((r1 // ((-2) + kernel_size1)) % ((-2) + kernel_size1))) +
                          kernel_size1 * kernel_size1 * (triton_helpers.div_floor_integer(r1, 4 + kernel_size1 * kernel_size1 + ((-4) * kernel_size1))) +
                          ((-4) * kernel_size1 * (triton_helpers.div_floor_integer(r1, 4 + kernel_size1 * kernel_size1 + ((-4) * kernel_size1)))) +
                          ((-2) * x0 * kernel_size1 * kernel_size1) + 4 * kernel_size0 * x0 + 8 * kernel_size1 * x0 +
                          kernel_size0 * x0 * kernel_size1 * kernel_size1 + ((-4) * kernel_size0 * kernel_size1 * x0) +
                          ((r1 % ((-2) + kernel_size1)))),
            r_mask & x_mask, eviction_policy='evict_last', other=0.0
        )

        zero = 0.0
        greater_than_zero = input_grad > zero
        equal_to_zero = input_grad == zero
        min_input_grad_zero = triton_helpers.minimum(input_grad, zero)
        min_input_grad_zero_ge_zero = min_input_grad_zero >= zero
        one = 1.0
        min_input_grad_zero_le_one = min_input_grad_zero <= one
        valid_range = min_input_grad_zero_ge_zero & min_input_grad_zero_le_one

        mask_float = mask_value.to(tl.float32)
        mask_scaled = mask_float * 1.25
        input_value_scaled = input_value * mask_scaled
        clamped_value = tl.where(valid_range, input_value_scaled, zero)
        half = 0.5
        half_clamped_value = clamped_value * half
        final_value = tl.where(equal_to_zero, half_clamped_value, clamped_value)
        selected_value = tl.where(greater_than_zero, zero, final_value)
        weighted_value = selected_value * weight_value

        temp_grad_sum_update = tl.broadcast_to(weighted_value, [XBLOCK, RBLOCK])
        temp_grad_sum = tl.where(r_mask & x_mask, temp_grad_sum + temp_grad_sum_update, temp_grad_sum)

        temp_weight_sum_update = tl.broadcast_to(selected_value, [XBLOCK, RBLOCK])
        temp_weight_sum = tl.where(r_mask & x_mask, temp_weight_sum + temp_weight_sum_update, temp_weight_sum)

    grad_sum = tl.sum(temp_grad_sum, 1)[:, None]
    weight_sum = tl.sum(temp_weight_sum, 1)[:, None]

    tl.store(output_grad_ptr0 + (x0), grad_sum, x_mask)
    tl.store(output_grad_ptr1 + (x0), weight_sum, x_mask)