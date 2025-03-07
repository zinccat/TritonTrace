# From: 89_ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_6(in_ptr0, out_ptr0, kernel_size0, kernel_size1, kernel_size2, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_num_elements = 2048
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_16 = input_index // 16
    input_index_mod_16 = input_index % 16
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_index_3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_2 = reduction_index
        temp0 = reduction_index_2 + input_index_16 * ((127 + 8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2) // 128)
        temp1 = 8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2
        temp2 = temp0 < temp1
        temp3 = tl.load(
            in_ptr0 + (
                8 * kernel_size1 * input_index_mod_16 * kernel_size2 * kernel_size2 +
                128 * kernel_size1 * kernel_size2 * kernel_size2 * (
                    ((reduction_index_2 + input_index_16 * ((127 + 8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2) // 128)) // (8 * kernel_size1 * kernel_size2 * kernel_size2)) % kernel_size0
                ) +
                ((reduction_index_2 + input_index_16 * ((127 + 8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2) // 128)) % (8 * kernel_size1 * kernel_size2 * kernel_size2))
            ),
            reduction_mask & temp2 & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        temp4 = tl.broadcast_to(temp3, [XBLOCK, RBLOCK])
        temp6 = temp_accumulator + temp4
        temp_accumulator = tl.where(reduction_mask & input_mask, temp6, temp_accumulator)

    temp5 = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(out_ptr0 + (input_index_3), temp5, input_mask)