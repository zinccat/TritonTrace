# From: 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool3d_2(input_ptr, output_ptr, kernel_size_z, kernel_size_y, kernel_size_x, stride_z, stride_y, stride_x, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements

    z_index = block_indices % kernel_size_z
    y_index = (block_indices // kernel_size_z) % kernel_size_y
    x_index = (block_indices // kernel_size_y) % kernel_size_x
    batch_index = block_indices // kernel_size_z
    linear_index = block_indices

    offset_base = (
        (-1) * batch_index +
        (-2) * y_index +
        2 * z_index +
        2 * x_index +
        (-8) * stride_x * x_index +
        (-4) * batch_index * stride_x * stride_x +
        2 * stride_z * batch_index +
        4 * stride_x * y_index +
        4 * stride_x * batch_index +
        8 * x_index * stride_x * stride_x +
        (-8) * stride_z * stride_x * batch_index +
        8 * stride_z * batch_index * stride_x * stride_x
    )

    tmp0 = tl.load(input_ptr + offset_base, valid_mask, eviction_policy='evict_last')
    tmp1 = tl.load(input_ptr + (1 + offset_base), valid_mask, eviction_policy='evict_last')
    tmp3 = tl.load(input_ptr + ((-1) + offset_base + 2 * stride_x), valid_mask, eviction_policy='evict_last')
    tmp5 = tl.load(input_ptr + (offset_base + 2 * stride_x), valid_mask, eviction_policy='evict_last')
    tmp7 = tl.load(input_ptr + (1 + offset_base + (-4) * stride_x), valid_mask, eviction_policy='evict_last')
    tmp9 = tl.load(input_ptr + (2 + offset_base + (-4) * stride_x), valid_mask, eviction_policy='evict_last')
    tmp11 = tl.load(input_ptr + (offset_base + (-2) * stride_x), valid_mask, eviction_policy='evict_last')
    tmp13 = tl.load(input_ptr + (1 + offset_base + (-2) * stride_x), valid_mask, eviction_policy='evict_last')

    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12

    avg_pool_factor = 0.125
    result = tmp14 * avg_pool_factor

    tl.store(output_ptr + linear_index, result, valid_mask)