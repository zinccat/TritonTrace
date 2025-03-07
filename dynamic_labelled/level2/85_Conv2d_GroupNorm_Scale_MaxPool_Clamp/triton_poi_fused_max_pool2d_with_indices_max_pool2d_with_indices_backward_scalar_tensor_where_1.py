# From: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_max_pool2d_with_indices_backward_scalar_tensor_where_1(
    input_ptr0, input_ptr1, output_ptr0, kernel_size0, kernel_size1, kernel_size2, total_elements, BLOCK_SIZE : tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < total_elements

    x_coord = block_indices % kernel_size0
    y_coord = (block_indices // kernel_size0) % kernel_size0
    z_coord = block_indices // kernel_size1
    linear_index = block_indices

    offset_y = (0 * (0 >= (y_coord // 2)) + (y_coord // 2) * ((y_coord // 2) > 0))
    offset_y_limit = (-1 + ((-1 + (kernel_size2 // 2)) * ((-1 + (kernel_size2 // 2)) <= (1 + (y_coord // 2))) + (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < (-1 + (kernel_size2 // 2)))))

    offset_x = (0 * (0 >= (x_coord // 2)) + (x_coord // 2) * ((x_coord // 2) > 0))
    offset_x_limit = (-1 + ((-1 + (kernel_size2 // 2)) * ((-1 + (kernel_size2 // 2)) <= (1 + (x_coord // 2))) + (1 + (x_coord // 2)) * ((1 + (x_coord // 2)) < (-1 + (kernel_size2 // 2)))))

    index_y = z_coord + ((-1) * (offset_y * (offset_y <= offset_y_limit) + (-1 + offset_y_limit) * (offset_y_limit < offset_y)))
    index_x = offset_x * (offset_x <= offset_x_limit) + (-1 + offset_x_limit) * (offset_x_limit < offset_x)

    tmp0 = tl.load(
        input_ptr0 + (index_y + index_x * kernel_size2 + z_coord * (kernel_size2 // 2) * (kernel_size2 // 2) + (kernel_size2 // 2) * offset_y + (-2) * z_coord * (kernel_size2 // 2)),
        valid_mask, eviction_policy='evict_last'
    )

    tmp12 = tl.load(
        input_ptr1 + (index_y + index_x * kernel_size2 + z_coord * (kernel_size2 // 2) * (kernel_size2 // 2) + (kernel_size2 // 2) * offset_y + (-2) * z_coord * (kernel_size2 // 2)),
        valid_mask, eviction_policy='evict_last'
    )

    divisor = tl.full([1], 2, tl.int32)
    tmp1 = tl.where((tmp0 < 0) != (divisor < 0), tl.where(tmp0 % divisor != 0, tmp0 // divisor - 1, tmp0 // divisor), tmp0 // divisor)
    tmp3 = tmp1 * divisor
    tmp4 = tmp0 - tmp3

    offset_y_double = 2 * (offset_y * (offset_y <= offset_y_limit) + (-1 + offset_y_limit) * (offset_y_limit < offset_y))
    tmp5 = offset_y_double
    tmp6 = tmp5 + tmp1

    offset_x_double = 2 * (offset_x * (offset_x <= offset_x_limit) + (-1 + offset_x_limit) * (offset_x_limit < offset_x))
    tmp7 = offset_x_double
    tmp8 = tmp7 + tmp4

    kernel_size0 = kernel_size0
    tmp9 = tmp6 * kernel_size0
    tmp10 = tmp9 + tmp8

    linear_index_input = block_indices % kernel_size1
    tmp11 = tmp10 == linear_index_input

    zero_value = 0.0
    tmp16 = tl.where(tmp11, tmp12, zero_value)

    tl.store(output_ptr0 + (linear_index), tmp16, valid_mask)