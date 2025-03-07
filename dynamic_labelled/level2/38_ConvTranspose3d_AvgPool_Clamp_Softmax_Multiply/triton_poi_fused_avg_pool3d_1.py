# From: 38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool3d_1poi_fused_avg_pool3d_1(input_ptr, output_ptr, kernel_size_dim0, kernel_size_dim1, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements

    kernel_index_dim0 = block_indices % kernel_size_dim0
    kernel_index_dim1 = (block_indices // kernel_size_dim0) % kernel_size_dim0
    kernel_index_dim2 = block_indices // kernel_size_dim1
    linear_index = block_indices

    value0 = tl.load(input_ptr + (2 * kernel_index_dim0 + 4 * kernel_size_dim0 * kernel_index_dim1 + 8 * kernel_index_dim2 * kernel_size_dim0 * kernel_size_dim0), valid_mask, eviction_policy='evict_last')
    value1 = tl.load(input_ptr + (1 + 2 * kernel_index_dim0 + 4 * kernel_size_dim0 * kernel_index_dim1 + 8 * kernel_size_dim1 * kernel_index_dim2), valid_mask, eviction_policy='evict_last')
    value3 = tl.load(input_ptr + (2 * kernel_size_dim0 + 2 * kernel_index_dim0 + 4 * kernel_size_dim0 * kernel_index_dim1 + 8 * kernel_size_dim1 * kernel_index_dim2), valid_mask, eviction_policy='evict_last')
    value5 = tl.load(input_ptr + (1 + 2 * kernel_size_dim0 + 2 * kernel_index_dim0 + 4 * kernel_size_dim0 * kernel_index_dim1 + 8 * kernel_size_dim1 * kernel_index_dim2), valid_mask, eviction_policy='evict_last')
    value7 = tl.load(input_ptr + (2 * kernel_index_dim0 + 4 * kernel_size_dim1 + 4 * kernel_size_dim0 * kernel_index_dim1 + 8 * kernel_size_dim1 * kernel_index_dim2), valid_mask, eviction_policy='evict_last')
    value9 = tl.load(input_ptr + (1 + 2 * kernel_index_dim0 + 4 * kernel_size_dim1 + 4 * kernel_size_dim0 * kernel_index_dim1 + 8 * kernel_size_dim1 * kernel_index_dim2), valid_mask, eviction_policy='evict_last')
    value11 = tl.load(input_ptr + (2 * kernel_size_dim0 + 2 * kernel_index_dim0 + 4 * kernel_size_dim1 + 4 * kernel_size_dim0 * kernel_index_dim1 + 8 * kernel_size_dim1 * kernel_index_dim2), valid_mask, eviction_policy='evict_last')
    value13 = tl.load(input_ptr + (1 + 2 * kernel_size_dim0 + 2 * kernel_index_dim0 + 4 * kernel_size_dim1 + 4 * kernel_size_dim0 * kernel_index_dim1 + 8 * kernel_size_dim1 * kernel_index_dim2), valid_mask, eviction_policy='evict_last')

    sum1 = value1 + value0
    sum2 = value3 + sum1
    sum3 = value5 + sum2
    sum4 = value7 + sum3
    sum5 = value9 + sum4
    sum6 = value11 + sum5
    sum7 = value13 + sum6

    avg_value = 0.125 * sum7

    tl.store(output_ptr + (linear_index), avg_value, valid_mask)