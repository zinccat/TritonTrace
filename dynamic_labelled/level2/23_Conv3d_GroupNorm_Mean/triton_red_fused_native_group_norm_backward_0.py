# From: 23_Conv3d_GroupNorm_Mean

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_0(
    input_grad_ptr, input_ptr, output_grad_ptr0, output_grad_ptr1, kernel_size0, kernel_size1, 
    x_num_elements, r_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_div_16 = x_index // 16
    input_grad = tl.load(input_grad_ptr + (x_div_16), x_mask, eviction_policy='evict_last')
    x_index_3 = x_index
    temp_sum0 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    temp_sum1 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r_index_2 = r_index
        input_value = tl.load(
            input_ptr + (
                (-8) * x_index_3 + 
                (-2) * ((r_index_2 // ((-2) + kernel_size1)) % ((-2) + kernel_size1)) + 
                4 * (triton_helpers.div_floor_integer(r_index_2, 4 + kernel_size1 * kernel_size1 + ((-4) * kernel_size1))) + 
                kernel_size1 * ((r_index_2 // ((-2) + kernel_size1)) % ((-2) + kernel_size1)) + 
                kernel_size1 * kernel_size1 * (triton_helpers.div_floor_integer(r_index_2, 4 + kernel_size1 * kernel_size1 + ((-4) * kernel_size1))) + 
                ((-4) * kernel_size1 * (triton_helpers.div_floor_integer(r_index_2, 4 + kernel_size1 * kernel_size1 + ((-4) * kernel_size1)))) + 
                ((-2) * x_index_3 * kernel_size1 * kernel_size1) + 
                4 * kernel_size0 * x_index_3 + 
                8 * kernel_size1 * x_index_3 + 
                kernel_size0 * x_index_3 * kernel_size1 * kernel_size1 + 
                ((-4) * kernel_size0 * kernel_size1 * x_index_3) + 
                (r_index_2 % ((-2) + kernel_size1))
            ), 
            r_mask & x_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        temp_const = (-128) + ((-32) * kernel_size1 * kernel_size1) + 64 * kernel_size0 + 128 * kernel_size1 + ((-64) * kernel_size0 * kernel_size1) + 16 * kernel_size0 * kernel_size1 * kernel_size1
        temp_const_float = temp_const.to(tl.float32)
        input_grad_normalized = input_grad / temp_const_float
        temp_product = input_grad_normalized * input_value
        temp_broadcast = tl.broadcast_to(temp_product, [XBLOCK, RBLOCK])
        temp_sum0_updated = temp_sum0 + temp_broadcast
        temp_sum0 = tl.where(r_mask & x_mask, temp_sum0_updated, temp_sum0)
        temp_broadcast1 = tl.broadcast_to(input_grad_normalized, [XBLOCK, RBLOCK])
        temp_sum1_updated = temp_sum1 + temp_broadcast1
        temp_sum1 = tl.where(r_mask & x_mask, temp_sum1_updated, temp_sum1)

    output_grad0 = tl.sum(temp_sum0, 1)[:, None]
    output_grad1 = tl.sum(temp_sum1, 1)[:, None]
    tl.store(output_grad_ptr0 + (x_index_3), output_grad0, x_mask)
    tl.store(output_grad_ptr1 + (x_index_3), output_grad1, x_mask)