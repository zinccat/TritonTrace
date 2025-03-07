# From: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool3d_6(input_ptr, output_ptr, kernel_size_z, kernel_size_y, kernel_size_x, kernel_size_total, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements

    z_index = block_indices % kernel_size_z
    y_index = (block_indices // kernel_size_z) % kernel_size_z
    x_index = (block_indices // kernel_size_y) % kernel_size_z
    batch_index = block_indices // kernel_size_x

    base_offset = batch_index * kernel_size_total * kernel_size_total * kernel_size_total
    y_offset = -4 * kernel_size_total * x_index
    z_offset = -3 * kernel_size_total * kernel_size_total * batch_index
    y_scale = 2 * kernel_size_total * y_index
    x_scale = 2 * x_index * kernel_size_total * kernel_size_total
    z_scale = 3 * kernel_size_total * batch_index

    offset = base_offset + y_offset + z_offset + y_scale + x_scale + z_scale

    tmp0 = tl.load(input_ptr + offset + (-1) * batch_index, valid_mask, eviction_policy='evict_last')
    tmp1 = tl.load(input_ptr + offset + 1 + (-1) * batch_index, valid_mask, eviction_policy='evict_last')
    tmp3 = tl.load(input_ptr + offset + (-1) + kernel_size_total + (-1) * batch_index, valid_mask, eviction_policy='evict_last')
    tmp5 = tl.load(input_ptr + offset + kernel_size_total + (-1) * batch_index, valid_mask, eviction_policy='evict_last')
    tmp7 = tl.load(input_ptr + offset + 1 + kernel_size_total * kernel_size_total + (-1) * batch_index + (-2) * kernel_size_total + (-2) * y_index, valid_mask, eviction_policy='evict_last')
    tmp9 = tl.load(input_ptr + offset + 2 + kernel_size_total * kernel_size_total + (-1) * batch_index + (-2) * kernel_size_total + (-2) * y_index, valid_mask, eviction_policy='evict_last')
    tmp11 = tl.load(input_ptr + offset + kernel_size_total * kernel_size_total + (-1) * kernel_size_total + (-1) * batch_index, valid_mask, eviction_policy='evict_last')
    tmp13 = tl.load(input_ptr + offset + 1 + kernel_size_total * kernel_size_total + (-1) * kernel_size_total + (-1) * batch_index, valid_mask, eviction_policy='evict_last')

    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12

    avg_factor = 0.125
    result = tmp14 * avg_factor

    output_offset = (z_index + y_index + x_index + batch_index +
                     y_index * (triton_helpers.div_floor_integer((-3) + kernel_size_total, 2)) +
                     x_index * (triton_helpers.div_floor_integer((-3) + kernel_size_total, 2)) * (triton_helpers.div_floor_integer((-3) + kernel_size_total, 2)) +
                     batch_index * (triton_helpers.div_floor_integer((-3) + kernel_size_total, 2)) * (triton_helpers.div_floor_integer((-3) + kernel_size_total, 2)) * (triton_helpers.div_floor_integer((-3) + kernel_size_total, 2)) +
                     2 * x_index * (triton_helpers.div_floor_integer((-3) + kernel_size_total, 2)) +
                     3 * batch_index * (triton_helpers.div_floor_integer((-3) + kernel_size_total, 2)) * (triton_helpers.div_floor_integer((-3) + kernel_size_total, 2)) +
                     3 * batch_index * (triton_helpers.div_floor_integer((-3) + kernel_size_total, 2)))

    tl.store(output_ptr + output_offset, result, valid_mask)