# From: 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool3d_2poi_fused_avg_pool3d_2(input_ptr, output_ptr, kernel_size_0, kernel_size_1, kernel_size_2, kernel_size_3, kernel_size_4, kernel_size_5, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_index = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_index < num_elements

    x_dim = block_index % kernel_size_0
    y_dim = (block_index // kernel_size_0) % kernel_size_0
    z_dim = (block_index // kernel_size_1) % kernel_size_2
    w_dim = block_index // kernel_size_3
    linear_index = block_index

    offset_base = (
        (-1) * w_dim +
        (-2) * y_dim +
        2 * x_dim +
        2 * z_dim +
        (-8) * kernel_size_5 * z_dim +
        (-4) * w_dim * kernel_size_5 * kernel_size_5 +
        2 * kernel_size_4 * w_dim +
        4 * kernel_size_5 * y_dim +
        4 * kernel_size_5 * w_dim +
        8 * z_dim * kernel_size_5 * kernel_size_5 +
        (-8) * kernel_size_4 * kernel_size_5 * w_dim +
        8 * kernel_size_4 * w_dim * kernel_size_5 * kernel_size_5
    )

    tmp0 = tl.load(input_ptr + offset_base, mask, eviction_policy='evict_last')
    tmp1 = tl.load(input_ptr + (1 + offset_base), mask, eviction_policy='evict_last')
    tmp3 = tl.load(input_ptr + ((-1) + offset_base + 2 * kernel_size_5), mask, eviction_policy='evict_last')
    tmp5 = tl.load(input_ptr + (offset_base + 2 * kernel_size_5), mask, eviction_policy='evict_last')
    tmp7 = tl.load(input_ptr + (1 + offset_base + (-4) * kernel_size_5 + 4 * kernel_size_5 * kernel_size_5), mask, eviction_policy='evict_last')
    tmp9 = tl.load(input_ptr + (2 + offset_base + (-4) * kernel_size_5 + 4 * kernel_size_5 * kernel_size_5), mask, eviction_policy='evict_last')
    tmp11 = tl.load(input_ptr + (offset_base + (-2) * kernel_size_5 + 4 * kernel_size_5 * kernel_size_5), mask, eviction_policy='evict_last')
    tmp13 = tl.load(input_ptr + (1 + offset_base + (-2) * kernel_size_5 + 4 * kernel_size_5 * kernel_size_5), mask, eviction_policy='evict_last')

    sum_tmp2 = tmp1 + tmp0
    sum_tmp4 = tmp3 + sum_tmp2
    sum_tmp6 = tmp5 + sum_tmp4
    sum_tmp8 = tmp7 + sum_tmp6
    sum_tmp10 = tmp9 + sum_tmp8
    sum_tmp12 = tmp11 + sum_tmp10
    sum_tmp14 = tmp13 + sum_tmp12

    avg_pool_factor = 0.125
    result = sum_tmp14 * avg_pool_factor

    tl.store(output_ptr + linear_index, result, mask)