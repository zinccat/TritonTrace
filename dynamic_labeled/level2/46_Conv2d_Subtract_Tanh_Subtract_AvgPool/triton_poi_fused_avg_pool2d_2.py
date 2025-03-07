# From: 46_Conv2d_Subtract_Tanh_Subtract_AvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool2d_2(in_ptr0, out_ptr0, kernel_size_0, kernel_size_1, kernel_size_2, num_elements, XBLOCK: tl.constexpr):
    program_id_offset = tl.program_id(0) * XBLOCK
    index_within_block = program_id_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = index_within_block < num_elements

    x_coord = index_within_block % kernel_size_0
    y_coord = (index_within_block // kernel_size_0) % kernel_size_0
    z_coord = index_within_block // kernel_size_1
    linear_index = index_within_block

    offset_base = z_coord * kernel_size_2 * kernel_size_2 + 2 * z_coord * kernel_size_2 + 2 * x_coord - 4 * y_coord

    load1 = tl.load(in_ptr0 + offset_base, valid_mask, eviction_policy='evict_last')
    load2 = tl.load(in_ptr0 + offset_base + 1, valid_mask, eviction_policy='evict_last')
    load3 = tl.load(in_ptr0 + offset_base + (-2) + kernel_size_2, valid_mask, eviction_policy='evict_last')
    load4 = tl.load(in_ptr0 + offset_base + (-1) + kernel_size_2, valid_mask, eviction_policy='evict_last')

    sum1 = load2 + load1
    sum2 = load3 + sum1
    sum3 = load4 + sum2

    avg_value = sum3 * 0.25

    tl.store(out_ptr0 + linear_index, avg_value, valid_mask)