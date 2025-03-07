# From: 55_Matmul_MaxPool_Sum_Scale

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_max_pool2d_with_indices_backward_0poi_fused_max_pool2d_with_indices_max_pool2d_with_indices_backward_0(input_ptr0, input_ptr1, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    index = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = index < num_elements
    col = index % 5
    row = index // 5
    linear_index = index

    # Load data from input_ptr0 with masking and eviction policy
    offset0 = 2 * row + (((0) * ((0) >= (col // 2)) + (col // 2) * ((col // 2) > (0)))) * ((((0) * ((0) >= (col // 2)) + (col // 2) * ((col // 2) > (0)))) <= ((-1) + ((2) * ((2) <= (1 + (col // 2))) + (1 + (col // 2)) * ((1 + (col // 2)) < (2))))) + ((-1) + ((2) * ((2) <= (1 + (col // 2))) + (1 + (col // 2)) * ((1 + (col // 2)) < (2)))) * (((-1) + ((2) * ((2) <= (1 + (col // 2))) + (1 + (col // 2)) * ((1 + (col // 2)) < (2)))) < (((0) * ((0) >= (col // 2)) + (col // 2) * ((col // 2) > (0)))))
    tmp0 = tl.load(input_ptr0 + offset0, mask, eviction_policy='evict_last')

    # Load data from input_ptr1 with masking and eviction policy
    tmp12 = tl.load(input_ptr1 + row, mask, eviction_policy='evict_last')

    # Calculate tmp1, tmp2, tmp3, tmp4
    tmp1 = tl.full([1], 2, tl.int32)
    tmp2 = tl.where((tmp0 < 0) != (tmp1 < 0), tl.where(tmp0 % tmp1 != 0, tmp0 // tmp1 - 1, tmp0 // tmp1), tmp0 // tmp1)
    tmp3 = tmp2 * tmp1
    tmp4 = tmp0 - tmp3

    # Calculate tmp5, tmp6
    tmp5 = tl.full([1], 0, tl.int64)
    tmp6 = tmp5 + tmp2

    # Calculate offset for tmp7, tmp8
    offset1 = 2 * (((0) * ((0) >= (col // 2)) + (col // 2) * ((col // 2) > (0)))) * ((((0) * ((0) >= (col // 2)) + (col // 2) * ((col // 2) > (0)))) <= ((-1) + ((2) * ((2) <= (1 + (col // 2))) + (1 + (col // 2)) * ((1 + (col // 2)) < (2))))) + ((-1) + ((2) * ((2) <= (1 + (col // 2))) + (1 + (col // 2)) * ((1 + (col // 2)) < (2)))) * (((-1) + ((2) * ((2) <= (1 + (col // 2))) + (1 + (col // 2)) * ((1 + (col // 2)) < (2)))) < (((0) * ((0) >= (col // 2)) + (col // 2) * ((col // 2) > (0)))))
    tmp7 = offset1 + tmp4
    tmp9 = tl.full([1], 5, tl.int64)
    tmp10 = tmp6 * tmp9
    tmp11 = tmp10 + tmp7

    # Calculate tmp14, tmp16, tmp18
    tmp13 = 0.5
    tmp14 = tmp12 * tmp13
    tmp15 = col
    tmp16 = tmp11 == tmp15
    tmp17 = 0.0
    tmp18 = tl.where(tmp16, tmp14, tmp17)

    # Store the result
    tl.store(output_ptr0 + linear_index, tmp18, mask)