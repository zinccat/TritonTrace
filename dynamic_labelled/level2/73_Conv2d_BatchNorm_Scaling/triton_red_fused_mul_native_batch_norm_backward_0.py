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
    input_block1 = input_index // 16
    input_block0 = (input_index % 16)
    temp_sum0 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_block3 = input_index
    temp_sum1 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_block2 = reduction_index
        temp_index0 = reduction_block2 + input_block1 * (triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15))
        temp_index1 = 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1
        temp_mask2 = temp_index0 < temp_index1

        temp_load0 = tl.load(
            input_grad_ptr + (
                (-2) * (((reduction_block2 + input_block1 * (triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)) // (-2 + kernel_size1)) % (-2 + kernel_size1)))
                + 4 * input_block0
                + 64 * (((reduction_block2 + input_block1 * (triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)) // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0))
                + kernel_size1 * (((reduction_block2 + input_block1 * (triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)) // (-2 + kernel_size1)) % (-2 + kernel_size1)))
                + input_block0 * kernel_size1 * kernel_size1
                + (-64) * kernel_size1 * (((reduction_block2 + input_block1 * (triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)) // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0))
                + (-4) * kernel_size1 * input_block0
                + 16 * kernel_size1 * kernel_size1 * (((reduction_block2 + input_block1 * (triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)) // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0))
                + ((reduction_block2 + input_block1 * (triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15))) % (-2 + kernel_size1))
            ),
            reduction_mask & temp_mask2 & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        temp_scaled0 = temp_load0 * 2.0
        temp_zeroed0 = tl.full(temp_scaled0.shape, 0, temp_scaled0.dtype)
        temp_masked0 = tl.where(temp_mask2, temp_scaled0, temp_zeroed0)
        temp_broadcasted0 = tl.broadcast_to(temp_masked0, [XBLOCK, RBLOCK])
        temp_accumulated0 = temp_sum0 + temp_broadcasted0
        temp_sum0 = tl.where(reduction_mask & input_mask, temp_accumulated0, temp_sum0)

        temp_load1 = tl.load(
            input_ptr + (
                (-2) * (((reduction_block2 + input_block1 * (triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)) // (-2 + kernel_size1)) % (-2 + kernel_size1)))
                + 4 * input_block0
                + 64 * (((reduction_block2 + input_block1 * (triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)) // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0))
                + kernel_size1 * (((reduction_block2 + input_block1 * (triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)) // (-2 + kernel_size1)) % (-2 + kernel_size1)))
                + input_block0 * kernel_size1 * kernel_size1
                + (-64) * kernel_size1 * (((reduction_block2 + input_block1 * (triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)) // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0))
                + (-4) * kernel_size1 * input_block0
                + 16 * kernel_size1 * kernel_size1 * (((reduction_block2 + input_block1 * (triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)) // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0))
                + ((reduction_block2 + input_block1 * (triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15))) % (-2 + kernel_size1))
            ),
            reduction_mask & temp_mask2 & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        temp_load2 = tl.load(
            scaling_factor_ptr + (tl.broadcast_to(input_block0, [XBLOCK, RBLOCK])),
            reduction_mask & temp_mask2 & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        temp_diff = temp_load1 - temp_load2
        temp_scaled1 = temp_scaled0 * temp_diff
        temp_zeroed1 = tl.full(temp_scaled1.shape, 0, temp_scaled1.dtype)
        temp_masked1 = tl.where(temp_mask2, temp_scaled1, temp_zeroed1)
        temp_broadcasted1 = tl.broadcast_to(temp_masked1, [XBLOCK, RBLOCK])
        temp_accumulated1 = temp_sum1 + temp_broadcasted1
        temp_sum1 = tl.where(reduction_mask & input_mask, temp_accumulated1, temp_sum1)

    output_grad0 = tl.sum(temp_sum0, 1)[:, None]
    output_grad1 = tl.sum(temp_sum1, 1)[:, None]
    tl.store(output_grad_ptr0 + (input_block3), output_grad0, input_mask)
    tl.store(output_grad_ptr1 + (input_block3), output_grad1, input_mask)