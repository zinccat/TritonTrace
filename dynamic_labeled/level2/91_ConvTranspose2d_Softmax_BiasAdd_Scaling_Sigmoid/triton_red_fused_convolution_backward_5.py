# From: 91_ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_5(input_ptr, output_ptr, kernel_size_0, kernel_size_1, kernel_size_2, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_num_elements = 448
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_1 = input_index // 64
    input_index_0 = (input_index % 64)
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_index_3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_2 = reduction_index

        temp_index_0 = reduction_index_2 + input_index_1 * ((6 + kernel_size_0 + 4 * kernel_size_0 * kernel_size_1 + 4 * kernel_size_0 * kernel_size_1 * kernel_size_1) // 7)
        temp_index_1 = kernel_size_0 + 4 * kernel_size_0 * kernel_size_1 + 4 * kernel_size_0 * kernel_size_1 * kernel_size_1
        temp_mask = temp_index_0 < temp_index_1

        loaded_value = tl.load(
            input_ptr + (
                input_index_0 + 64 * (
                    ((temp_index_0 // kernel_size_2) % kernel_size_0)
                ) + 2 * kernel_size_1 * (
                    ((temp_index_0 // (1 + 2 * kernel_size_1)) % (1 + 2 * kernel_size_1))
                ) + 4 * kernel_size_1 * input_index_0 + 4 * input_index_0 * kernel_size_1 * kernel_size_1 + 256 * kernel_size_1 * (
                    ((temp_index_0 // kernel_size_2) % kernel_size_0)
                ) + 256 * kernel_size_1 * kernel_size_1 * (
                    ((temp_index_0 // kernel_size_2) % kernel_size_0)
                ) + ((temp_index_0 % (1 + 2 * kernel_size_1))) + (
                    ((temp_index_0 // (1 + 2 * kernel_size_1)) % (1 + 2 * kernel_size_1))
                )
            ),
            reduction_mask & temp_mask & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        broadcasted_value = tl.broadcast_to(loaded_value, [XBLOCK, RBLOCK])
        temp_accumulator_updated = temp_accumulator + broadcasted_value
        temp_accumulator = tl.where(reduction_mask & input_mask, temp_accumulator_updated, temp_accumulator)

    summed_result = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (input_index_3), summed_result, input_mask)