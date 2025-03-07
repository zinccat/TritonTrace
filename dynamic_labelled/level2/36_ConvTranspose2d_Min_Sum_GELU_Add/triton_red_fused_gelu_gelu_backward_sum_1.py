# From: 36_ConvTranspose2d_Min_Sum_GELU_Add

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_gelu_gelu_backward_sum_1red_fused_gelu_gelu_backward_sum_1(
    input_ptr, output_ptr0, output_ptr1, kernel_size0, kernel_size1, input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_x0 = (input_index % kernel_size0)
    input_x1 = input_index // kernel_size0
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_x3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_r2 = reduction_index
        temp0 = tl.load(
            input_ptr + (input_x0 + 2 * kernel_size1 * reduction_r2 + 4 * input_x1 * kernel_size1 * kernel_size1), 
            reduction_mask & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        temp1 = tl.broadcast_to(temp0, [XBLOCK, RBLOCK])
        temp3 = temp_accumulator + temp1
        temp_accumulator = tl.where(reduction_mask & input_mask, temp3, temp_accumulator)

    temp2 = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr0 + (input_x3), temp2, input_mask)

    sqrt_half = 0.7071067811865476
    temp5 = temp2 * sqrt_half
    erf_result = tl.extra.cuda.libdevice.erf(temp5)
    one = 1.0
    temp8 = erf_result + one
    half = 0.5
    temp10 = temp8 * half
    temp11 = temp2 * temp2
    neg_half = -0.5
    temp13 = temp11 * neg_half
    exp_result = tl.math.exp(temp13)
    sqrt_two_pi = 0.3989422804014327
    temp16 = exp_result * sqrt_two_pi
    temp17 = temp2 * temp16
    final_result = temp10 + temp17

    tl.store(output_ptr1 + (input_x3), final_result, input_mask)