# From: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool3d_5poi_fused_avg_pool3d_5(input_ptr, output_ptr, kernel_size_z, kernel_size_y, kernel_size_x, kernel_size_w, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements

    z_index = block_indices % kernel_size_z
    y_index = (block_indices // kernel_size_z) % kernel_size_z
    x_index = (block_indices // kernel_size_y) % kernel_size_z
    w_index = block_indices // kernel_size_x
    linear_index = block_indices

    offset_base = (-1 * w_index) + (-2 * y_index) + 2 * z_index + 2 * x_index + (-12 * w_index * kernel_size_w * kernel_size_w) + (-8 * kernel_size_w * x_index) + 4 * kernel_size_w * y_index + 6 * kernel_size_w * w_index + 8 * x_index * kernel_size_w * kernel_size_w + 8 * w_index * kernel_size_w * kernel_size_w * kernel_size_w

    tmp0 = tl.load(input_ptr + offset_base, valid_mask, eviction_policy='evict_last')
    tmp1 = tl.load(input_ptr + (1 + offset_base), valid_mask, eviction_policy='evict_last')
    tmp3 = tl.load(input_ptr + ((-1) + offset_base + 2 * kernel_size_w), valid_mask, eviction_policy='evict_last')
    tmp5 = tl.load(input_ptr + (offset_base + 2 * kernel_size_w), valid_mask, eviction_policy='evict_last')
    tmp7 = tl.load(input_ptr + (1 + offset_base + (-4) * kernel_size_w), valid_mask, eviction_policy='evict_last')
    tmp9 = tl.load(input_ptr + (2 + offset_base + (-4) * kernel_size_w), valid_mask, eviction_policy='evict_last')
    tmp11 = tl.load(input_ptr + (offset_base + (-2) * kernel_size_w), valid_mask, eviction_policy='evict_last')
    tmp13 = tl.load(input_ptr + (1 + offset_base + (-2) * kernel_size_w), valid_mask, eviction_policy='evict_last')

    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12

    avg_factor = 0.125
    result = tmp14 * avg_factor

    tl.store(output_ptr + (linear_index), result, valid_mask)