# From: 38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool3d_1(in_ptr0, out_ptr0, kernel_size_z, kernel_size_y, total_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < total_elements
    z = index % kernel_size_z
    y = (index // kernel_size_z) % kernel_size_z
    x = index // (kernel_size_y * kernel_size_z)
    linear_index = index

    # Load neighboring elements for averaging
    neighbor_0 = tl.load(in_ptr0 + (2*z + 4*kernel_size_z*y + 8*x*kernel_size_z*kernel_size_z), mask, eviction_policy='evict_last')
    neighbor_1 = tl.load(in_ptr0 + (1 + 2*z + 4*kernel_size_z*y + 8*kernel_size_y*x), mask, eviction_policy='evict_last')
    neighbor_3 = tl.load(in_ptr0 + (2*kernel_size_z + 2*z + 4*kernel_size_z*y + 8*kernel_size_y*x), mask, eviction_policy='evict_last')
    neighbor_5 = tl.load(in_ptr0 + (1 + 2*kernel_size_z + 2*z + 4*kernel_size_z*y + 8*kernel_size_y*x), mask, eviction_policy='evict_last')
    neighbor_7 = tl.load(in_ptr0 + (2*z + 4*kernel_size_y + 4*kernel_size_z*y + 8*kernel_size_y*x), mask, eviction_policy='evict_last')
    neighbor_9 = tl.load(in_ptr0 + (1 + 2*z + 4*kernel_size_y + 4*kernel_size_z*y + 8*kernel_size_y*x), mask, eviction_policy='evict_last')
    neighbor_11 = tl.load(in_ptr0 + (2*kernel_size_z + 2*z + 4*kernel_size_y + 4*kernel_size_z*y + 8*kernel_size_y*x), mask, eviction_policy='evict_last')
    neighbor_13 = tl.load(in_ptr0 + (1 + 2*kernel_size_z + 2*z + 4*kernel_size_y + 4*kernel_size_z*y + 8*kernel_size_y*x), mask, eviction_policy='evict_last')

    # Sum the neighbors
    sum_2 = neighbor_1 + neighbor_0
    sum_4 = neighbor_3 + sum_2
    sum_6 = neighbor_5 + sum_4
    sum_8 = neighbor_7 + sum_6
    sum_10 = neighbor_9 + sum_8
    sum_12 = neighbor_11 + sum_10
    sum_14 = neighbor_13 + sum_12

    # Average the sum
    average = 0.125
    result = sum_14 * average

    # Store the result
    tl.store(out_ptr0 + (linear_index), result, mask)