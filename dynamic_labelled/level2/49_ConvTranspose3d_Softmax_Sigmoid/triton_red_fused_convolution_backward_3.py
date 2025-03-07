# From: 49_ConvTranspose3d_Softmax_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_3(in_ptr0, out_ptr0, kernel_size0, kernel_size1, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_num_elements = 1024
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
            in_ptr0 + (
                64 * (((reduction_r2 + 512 * input_x1 * kernel_size1 * kernel_size1) // 64) % (128 * kernel_size1))
                + 8192 * kernel_size1 * input_x0
                + 524288 * kernel_size1 * (((reduction_r2 + 512 * input_x1 * kernel_size1 * kernel_size1) // kernel_size0) % kernel_size1)
                + (reduction_r2 % 64)
            ),
            reduction_mask & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        temp1 = tl.broadcast_to(temp0, [XBLOCK, RBLOCK])
        temp3 = temp_accumulator + temp1
        temp_accumulator = tl.where(reduction_mask & input_mask, temp3, temp_accumulator)

    temp2 = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(out_ptr0 + (input_x3), temp2, input_mask)