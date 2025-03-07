# From: 10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_hardtanh_backward_max_pool2d_with_indices_max_pool2d_with_indices_backward_1poi_fused_div_hardtanh_backward_max_pool2d_with_indices_max_pool2d_with_indices_backward_1(
    input_ptr0, input_ptr1, output_ptr0, kernel_size_0, kernel_size_1, kernel_size_2, kernel_size_3, total_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < total_elements

    x_index_0 = block_indices % kernel_size_0
    x_index_1 = (block_indices // kernel_size_0) % kernel_size_0
    x_index_2 = block_indices // kernel_size_1
    x_index_5 = block_indices % kernel_size_1
    x_index_4 = block_indices

    load_index_0 = (
        kernel_size_2 * x_index_2 +
        kernel_size_3 * (
            ((0) * ((0) >= (x_index_1 // 2)) + (x_index_1 // 2) * ((x_index_1 // 2) > (0))) *
            (((0) * ((0) >= (x_index_1 // 2)) + (x_index_1 // 2) * ((x_index_1 // 2) > (0))) <=
             ((-1) + (kernel_size_3 * (kernel_size_3 <= (1 + (x_index_1 // 2))) + (1 + (x_index_1 // 2)) * ((1 + (x_index_1 // 2)) < kernel_size_3))) +
             ((-1) + (kernel_size_3 * (kernel_size_3 <= (1 + (x_index_1 // 2))) + (1 + (x_index_1 // 2)) * ((1 + (x_index_1 // 2)) < kernel_size_3))) *
             (((-1) + (kernel_size_3 * (kernel_size_3 <= (1 + (x_index_1 // 2))) + (1 + (x_index_1 // 2)) * ((1 + (x_index_1 // 2)) < kernel_size_3))) < 
              ((0) * ((0) >= (x_index_1 // 2)) + (x_index_1 // 2) * ((x_index_1 // 2) > (0))))
        ) +
        (
            ((0) * ((0) >= (x_index_0 // 2)) + (x_index_0 // 2) * ((x_index_0 // 2) > (0))) *
            (((0) * ((0) >= (x_index_0 // 2)) + (x_index_0 // 2) * ((x_index_0 // 2) > (0))) <=
             ((-1) + (kernel_size_3 * (kernel_size_3 <= (1 + (x_index_0 // 2))) + (1 + (x_index_0 // 2)) * ((1 + (x_index_0 // 2)) < kernel_size_3))) +
             ((-1) + (kernel_size_3 * (kernel_size_3 <= (1 + (x_index_0 // 2))) + (1 + (x_index_0 // 2)) * ((1 + (x_index_0 // 2)) < kernel_size_3))) *
             (((-1) + (kernel_size_3 * (kernel_size_3 <= (1 + (x_index_0 // 2))) + (1 + (x_index_0 // 2)) * ((1 + (x_index_0 // 2)) < kernel_size_3))) < 
              ((0) * ((0) >= (x_index_0 // 2)) + (x_index_0 // 2) * ((x_index_0 // 2) > (0))))
        )
    )

    load_index_1 = load_index_0

    input_value_0 = tl.load(input_ptr0 + load_index_0, valid_mask, eviction_policy='evict_last')
    input_value_1 = tl.load(input_ptr1 + load_index_1, valid_mask, eviction_policy='evict_last')

    divisor = tl.full([1], 2, tl.int32)
    quotient = tl.where((input_value_0 < 0) != (divisor < 0), tl.where(input_value_0 % divisor != 0, input_value_0 // divisor - 1, input_value_0 // divisor), input_value_0 // divisor)
    product = quotient * divisor
    remainder = input_value_0 - product

    index_1_adjustment = 2 * (
        ((0) * ((0) >= (x_index_1 // 2)) + (x_index_1 // 2) * ((x_index_1 // 2) > (0))) *
        (((0) * ((0) >= (x_index_1 // 2)) + (x_index_1 // 2) * ((x_index_1 // 2) > (0))) <=
         ((-1) + (kernel_size_3 * (kernel_size_3 <= (1 + (x_index_1 // 2))) + (1 + (x_index_1 // 2)) * ((1 + (x_index_1 // 2)) < kernel_size_3))) +
         ((-1) + (kernel_size_3 * (kernel_size_3 <= (1 + (x_index_1 // 2))) + (1 + (x_index_1 // 2)) * ((1 + (x_index_1 // 2)) < kernel_size_3))) *
         (((-1) + (kernel_size_3 * (kernel_size_3 <= (1 + (x_index_1 // 2))) + (1 + (x_index_1 // 2)) * ((1 + (x_index_1 // 2)) < kernel_size_3))) < 
          ((0) * ((0) >= (x_index_1 // 2)) + (x_index_1 // 2) * ((x_index_1 // 2) > (0))))
    )
    adjusted_index_1 = index_1_adjustment + quotient

    index_0_adjustment = 2 * (
        ((0) * ((0) >= (x_index_0 // 2)) + (x_index_0 // 2) * ((x_index_0 // 2) > (0))) *
        (((0) * ((0) >= (x_index_0 // 2)) + (x_index_0 // 2) * ((x_index_0 // 2) > (0))) <=
         ((-1) + (kernel_size_3 * (kernel_size_3 <= (1 + (x_index_0 // 2))) + (1 + (x_index_0 // 2)) * ((1 + (x_index_0 // 2)) < kernel_size_3))) +
         ((-1) + (kernel_size_3 * (kernel_size_3 <= (1 + (x_index_0 // 2))) + (1 + (x_index_0 // 2)) * ((1 + (x_index_0 // 2)) < kernel_size_3))) *
         (((-1) + (kernel_size_3 * (kernel_size_3 <= (1 + (x_index_0 // 2))) + (1 + (x_index_0 // 2)) * ((1 + (x_index_0 // 2)) < kernel_size_3))) < 
          ((0) * ((0) >= (x_index_0 // 2)) + (x_index_0 // 2) * ((x_index_0 // 2) > (0))))
    )
    adjusted_index_0 = index_0_adjustment + remainder

    final_index = adjusted_index_1 * kernel_size_0 + adjusted_index_0

    comparison_index = x_index_5
    condition_met = final_index == comparison_index

    output_value = tl.where(condition_met, input_value_1, 0.0)

    tl.store(output_ptr0 + x_index_4, output_value, valid_mask)