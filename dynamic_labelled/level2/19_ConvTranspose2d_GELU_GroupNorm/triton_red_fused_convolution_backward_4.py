# From: 19_ConvTranspose2d_GELU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_4red_fused_convolution_backward_4(
    input_ptr, output_ptr, kernel_size, input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 384
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_x0 = (input_index % 64)
    input_x1 = input_index // 64
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_x3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_r2 = reduction_index
        temp0 = tl.load(
            input_ptr + (
                66 * (((reduction_r2 + 726 * kernel_size * input_x1) // 66) % 66) 
                + 4356 * input_x0 
                + 278784 * (((reduction_r2 + 726 * kernel_size * input_x1) // 4356) % kernel_size) 
                + (reduction_r2 % 66)
            ), 
            reduction_mask & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        temp1 = tl.broadcast_to(temp0, [XBLOCK, RBLOCK])
        temp3 = temp_accumulator + temp1
        temp_accumulator = tl.where(reduction_mask & input_mask, temp3, temp_accumulator)

    temp2 = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (input_x3), temp2, input_mask)