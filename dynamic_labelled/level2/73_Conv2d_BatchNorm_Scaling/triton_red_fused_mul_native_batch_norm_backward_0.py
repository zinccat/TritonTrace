# From: 73_Conv2d_BatchNorm_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mul_native_batch_norm_backward_0(
    input_grad_ptr, input_ptr, scaling_factor_ptr, output_grad_ptr0, output_grad_ptr1, kernel_size0, kernel_size1, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 240
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_16 = input_index // 16
    input_index_mod_16 = input_index % 16
    temp_sum0 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_index_flat = input_index
    temp_sum1 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_flat = reduction_index
        temp_index = reduction_index_flat + input_index_16 * (
            triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
        )
        temp_limit = 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1
        temp_condition = temp_index < temp_limit
        temp_load = tl.load(
            input_grad_ptr + (
                (-2) * (((reduction_index_flat + input_index_16 * (
                    triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
                )) // ((-2) + kernel_size1)) % ((-2) + kernel_size1))
                + 4 * input_index_mod_16
                + 64 * (((reduction_index_flat + input_index_16 * (
                    triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
                )) // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0)
                + kernel_size1 * (((reduction_index_flat + input_index_16 * (
                    triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
                )) // ((-2) + kernel_size1)) % ((-2) + kernel_size1))
                + input_index_mod_16 * kernel_size1 * kernel_size1
                + (-64) * kernel_size1 * (((reduction_index_flat + input_index_16 * (
                    triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
                )) // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0)
                + (-4) * kernel_size1 * input_index_mod_16
                + 16 * kernel_size1 * kernel_size1 * (((reduction_index_flat + input_index_16 * (
                    triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
                )) // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0)
                + ((reduction_index_flat + input_index_16 * (
                    triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
                )) % ((-2) + kernel_size1))
            ),
            reduction_mask & temp_condition & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        temp_scaled = temp_load * 2.0
        temp_zeroed = tl.full(temp_scaled.shape, 0, temp_scaled.dtype)
        temp_selected = tl.where(temp_condition, temp_scaled, temp_zeroed)
        temp_broadcasted = tl.broadcast_to(temp_selected, [XBLOCK, RBLOCK])
        temp_accumulated0 = temp_sum0 + temp_broadcasted
        temp_sum0 = tl.where(reduction_mask & input_mask, temp_accumulated0, temp_sum0)

        temp_load1 = tl.load(
            input_ptr + (
                (-2) * (((reduction_index_flat + input_index_16 * (
                    triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
                )) // ((-2) + kernel_size1)) % ((-2) + kernel_size1))
                + 4 * input_index_mod_16
                + 64 * (((reduction_index_flat + input_index_16 * (
                    triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
                )) // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0)
                + kernel_size1 * (((reduction_index_flat + input_index_16 * (
                    triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
                )) // ((-2) + kernel_size1)) % ((-2) + kernel_size1))
                + input_index_mod_16 * kernel_size1 * kernel_size1
                + (-64) * kernel_size1 * (((reduction_index_flat + input_index_16 * (
                    triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
                )) // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0)
                + (-4) * kernel_size1 * input_index_mod_16
                + 16 * kernel_size1 * kernel_size1 * (((reduction_index_flat + input_index_16 * (
                    triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
                )) // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0)
                + ((reduction_index_flat + input_index_16 * (
                    triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
                )) % ((-2) + kernel_size1))
            ),
            reduction_mask & temp_condition & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        temp_load2 = tl.load(
            scaling_factor_ptr + (tl.broadcast_to(input_index_mod_16, [XBLOCK, RBLOCK])),
            reduction_mask & temp_condition & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        temp_diff = temp_load1 - temp_load2
        temp_product = temp_scaled * temp_diff
        temp_zeroed1 = tl.full(temp_product.shape, 0, temp_product.dtype)
        temp_selected1 = tl.where(temp_condition, temp_product, temp_zeroed1)
        temp_broadcasted1 = tl.broadcast_to(temp_selected1, [XBLOCK, RBLOCK])
        temp_accumulated1 = temp_sum1 + temp_broadcasted1
        temp_sum1 = tl.where(reduction_mask & input_mask, temp_accumulated1, temp_sum1)

    temp_sum0_final = tl.sum(temp_sum0, 1)[:, None]
    temp_sum1_final = tl.sum(temp_sum1, 1)[:, None]
    tl.store(output_grad_ptr0 + (input_index_flat), temp_sum0_final, input_mask)
    tl.store(output_grad_ptr1 + (input_index_flat), temp_sum1_final, input_mask)