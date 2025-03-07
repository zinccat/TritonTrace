# From: 38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax__softmax_backward_data_avg_pool3d_backward_clamp_ge_le_logical_and_mul_scalar_tensor_where_2poi_fused__softmax__softmax_backward_data_avg_pool3d_backward_clamp_ge_le_logical_and_mul_scalar_tensor_where_2(
    input_ptr, output_ptr, kernel_size_d, kernel_size_h, kernel_size_w, stride_d, stride_h, stride_w, num_elements, BLOCK_SIZE : tl.constexpr):

    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements

    index_d = block_indices % kernel_size_d
    index_h = (block_indices // kernel_size_d) % kernel_size_h
    index_w = (block_indices // kernel_size_w) % kernel_size_w
    index_n = block_indices // stride_d
    linear_index = block_indices

    input_index = (
        kernel_size_w * (
            (0 * (0 >= (index_h // 2)) + (index_h // 2) * ((index_h // 2) > 0)) *
            ((0 * (0 >= (index_h // 2)) + (index_h // 2) * ((index_h // 2) > 0)) <=
             (-1 + (kernel_size_w * (kernel_size_w <= (1 + (index_h // 2))) + (1 + (index_h // 2)) * ((1 + (index_h // 2)) < kernel_size_w))))
            + (-1 + (kernel_size_w * (kernel_size_w <= (1 + (index_h // 2))) + (1 + (index_h // 2)) * ((1 + (index_h // 2)) < kernel_size_w))) *
            ((-1 + (kernel_size_w * (kernel_size_w <= (1 + (index_h // 2))) + (1 + (index_h // 2)) * ((1 + (index_h // 2)) < kernel_size_w))) < 
             (0 * (0 >= (index_h // 2)) + (index_h // 2) * ((index_h // 2) > 0)))
        ) +
        kernel_size_w * kernel_size_h * (
            (0 * (0 >= (index_w // 2)) + (index_w // 2) * ((index_w // 2) > 0)) *
            ((0 * (0 >= (index_w // 2)) + (index_w // 2) * ((index_w // 2) > 0)) <=
             (-1 + (kernel_size_h * (kernel_size_h <= (1 + (index_w // 2))) + (1 + (index_w // 2)) * ((1 + (index_w // 2)) < kernel_size_h))))
            + (-1 + (kernel_size_h * (kernel_size_h <= (1 + (index_w // 2))) + (1 + (index_w // 2)) * ((1 + (index_w // 2)) < kernel_size_h))) *
            ((-1 + (kernel_size_h * (kernel_size_h <= (1 + (index_w // 2))) + (1 + (index_w // 2)) * ((1 + (index_w // 2)) < kernel_size_h))) < 
             (0 * (0 >= (index_w // 2)) + (index_w // 2) * ((index_w // 2) > 0)))
        ) +
        kernel_size_h * index_n * kernel_size_w * kernel_size_w +
        (0 * (0 >= (index_d // 2)) + (index_d // 2) * ((index_d // 2) > 0)) *
        ((0 * (0 >= (index_d // 2)) + (index_d // 2) * ((index_d // 2) > 0)) <=
         (-1 + (kernel_size_w * (kernel_size_w <= (1 + (index_d // 2))) + (1 + (index_d // 2)) * ((1 + (index_d // 2)) < kernel_size_w))))
        + (-1 + (kernel_size_w * (kernel_size_w <= (1 + (index_d // 2))) + (1 + (index_d // 2)) * ((1 + (index_d // 2)) < kernel_size_w))) *
        ((-1 + (kernel_size_w * (kernel_size_w <= (1 + (index_d // 2))) + (1 + (index_d // 2)) * ((1 + (index_d // 2)) < kernel_size_w))) < 
         (0 * (0 >= (index_d // 2)) + (index_d // 2) * ((index_d // 2) > 0)))
    )

    loaded_value = tl.load(input_ptr + input_index, valid_mask, eviction_policy='evict_last')
    averaged_value = loaded_value / 8

    condition_h = (0 * (0 >= (index_h // 2)) + (index_h // 2) * ((index_h // 2) > 0)) < ((kernel_size_w * (kernel_size_w <= (1 + (index_h // 2))) + (1 + (index_h // 2)) * ((1 + (index_h // 2)) < kernel_size_w)))
    condition_w = (0 * (0 >= (index_w // 2)) + (index_w // 2) * ((index_w // 2) > 0)) < ((kernel_size_h * (kernel_size_h <= (1 + (index_w // 2))) + (1 + (index_w // 2)) * ((1 + (index_w // 2)) < kernel_size_h)))
    condition_d = (0 * (0 >= (index_d // 2)) + (index_d // 2) * ((index_d // 2) > 0)) < ((kernel_size_w * (kernel_size_w <= (1 + (index_d // 2))) + (1 + (index_d // 2)) * ((1 + (index_d // 2)) < kernel_size_w)))

    combined_condition = condition_h & condition_w & condition_d

    result_value = tl.where(combined_condition, averaged_value, 0.0)
    tl.store(output_ptr + linear_index, result_value, valid_mask)