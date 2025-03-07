# From: 97_Matmul_BatchNorm_BiasAdd_Divide_Swish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_div_mul_sigmoid_sigmoid_backward_sum_1red_fused_add_div_mul_sigmoid_sigmoid_backward_sum_1(
    input_ptr0, input_ptr1, output_ptr0, kernel_size, total_elements, reduction_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    total_elements = 8
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_0 = x_indices
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_elements
        r_indices_1 = r_indices
        temp0 = tl.load(
            input_ptr0 + (((r_indices_1 + 64 * kernel_size * x_indices_0) % (512 * kernel_size))),
            r_mask & x_mask, eviction_policy='evict_last', other=0.0
        )
        temp1 = tl.load(
            input_ptr1 + (((r_indices_1 + 64 * kernel_size * x_indices_0) % (512 * kernel_size))),
            r_mask & x_mask, eviction_policy='evict_last', other=0.0
        )
        sigmoid_temp1 = tl.sigmoid(temp1)
        temp3 = temp0 * sigmoid_temp1
        temp4 = temp0 * temp1
        one = 1.0
        temp6 = one - sigmoid_temp1
        temp7 = sigmoid_temp1 * temp6
        temp8 = temp4 * temp7
        temp9 = temp3 + temp8
        temp10 = temp9 * one
        temp11 = tl.broadcast_to(temp10, [XBLOCK, RBLOCK])
        temp13 = temp_accumulator + temp11
        temp_accumulator = tl.where(r_mask & x_mask, temp13, temp_accumulator)

    temp12 = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr0 + (x_indices_0), temp12, x_mask)