# From: 6_Conv3d_Softmax_MaxPool_MaxPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_backward_data_0per_fused__softmax_backward_data_0(
    input_grad_ptr, output_grad_ptr, output_ptr, kernel_size_0, kernel_size_1, kernel_size_2, 
    input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < input_num_elements
    r_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_index
    x3 = (x_index % kernel_size_0)
    x4 = x_index // kernel_size_0
    x5 = x_index

    input_grad = tl.load(
        input_grad_ptr + (
            x3 + ((-128) * x4) + ((-8) * r2) + 
            ((-32) * x4 * kernel_size_2 * kernel_size_2) + 
            ((-2) * r2 * kernel_size_2 * kernel_size_2) + 
            4 * kernel_size_1 * r2 + 
            8 * kernel_size_2 * r2 + 
            64 * kernel_size_1 * x4 + 
            128 * kernel_size_2 * x4 + 
            kernel_size_1 * r2 * kernel_size_2 * kernel_size_2 + 
            ((-64) * kernel_size_1 * kernel_size_2 * x4) + 
            ((-4) * kernel_size_1 * kernel_size_2 * r2) + 
            16 * kernel_size_1 * x4 * kernel_size_2 * kernel_size_2
        ), 
        x_mask, 
        eviction_policy='evict_last', 
        other=0.0
    )

    output_grad = tl.load(
        output_grad_ptr + (
            x3 + ((-128) * x4) + ((-8) * r2) + 
            ((-32) * x4 * kernel_size_2 * kernel_size_2) + 
            ((-2) * r2 * kernel_size_2 * kernel_size_2) + 
            4 * kernel_size_1 * r2 + 
            8 * kernel_size_2 * r2 + 
            64 * kernel_size_1 * x4 + 
            128 * kernel_size_2 * x4 + 
            kernel_size_1 * r2 * kernel_size_2 * kernel_size_2 + 
            ((-64) * kernel_size_1 * kernel_size_2 * x4) + 
            ((-4) * kernel_size_1 * kernel_size_2 * r2) + 
            16 * kernel_size_1 * x4 * kernel_size_2 * kernel_size_2
        ), 
        x_mask, 
        eviction_policy='evict_last', 
        other=0.0
    )

    element_wise_product = input_grad * output_grad
    broadcasted_product = tl.broadcast_to(element_wise_product, [XBLOCK, RBLOCK])
    masked_product = tl.where(x_mask, broadcasted_product, 0)
    sum_over_reduction = tl.sum(masked_product, 1)[:, None]

    tl.store(output_ptr + (x5), sum_over_reduction, x_mask)