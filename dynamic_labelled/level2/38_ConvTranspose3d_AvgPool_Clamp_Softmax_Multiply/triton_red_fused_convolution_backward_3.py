# From: 38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_3(in_ptr0, out_ptr0, kernel_size0, kernel_size1, kernel_size2, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_num_elements = 336
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_dim0 = (input_index % 21)
    input_dim1 = input_index // 21
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_flat_index = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_dim0 = reduction_index

        temp_index0 = reduction_dim0 + input_dim0 * ((20 + 8 * kernel_size0 * kernel_size0 * kernel_size1 * kernel_size1) // 21)
        temp_index1 = 8 * kernel_size0 * kernel_size0 * kernel_size1 * kernel_size1
        temp_mask = temp_index0 < temp_index1

        temp_load = tl.load(
            in_ptr0 + (
                8 * kernel_size0 * input_dim1 * kernel_size1 * kernel_size1 +
                128 * kernel_size0 * kernel_size1 * kernel_size1 * (
                    ((temp_index0 // kernel_size2) % kernel_size0
                ) +
                (temp_index0 % kernel_size2)
            ),
            reduction_mask & temp_mask & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        temp_broadcast = tl.broadcast_to(temp_load, [XBLOCK, RBLOCK])
        temp_sum = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(reduction_mask & input_mask, temp_sum, temp_accumulator)

    temp_result = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(out_ptr0 + (input_flat_index), temp_result, input_mask)