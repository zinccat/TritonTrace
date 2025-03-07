# From: 46_Conv2d_Subtract_Tanh_Subtract_AvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool2d_backward_sub_tanh_tanh_backward_0poi_fused_avg_pool2d_backward_sub_tanh_tanh_backward_0(
    in_out_ptr, input_ptr, kernel_size_0, kernel_size_1, kernel_size_2, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    x_coord = index % kernel_size_0
    y_coord = (index // kernel_size_0) % kernel_size_0
    z_coord = index // kernel_size_1
    linear_index = index

    # Load input with complex indexing
    input_index = (
        z_coord
        + ((-1) * (
            (0 if (0 >= (y_coord // 2)) else (y_coord // 2))
            * (
                (0 if (0 >= (y_coord // 2)) else (y_coord // 2)) <=
                ((-1) + (((-1) + (kernel_size_2 // 2))
                         * (((-1) + (kernel_size_2 // 2)) <= (1 + (y_coord // 2)))
                         + (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size_2 // 2))))
            )
            + ((-1) + (((-1) + (kernel_size_2 // 2))
                       * (((-1) + (kernel_size_2 // 2)) <= (1 + (y_coord // 2)))
                       + (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size_2 // 2))))
            * (((-1) + (((-1) + (kernel_size_2 // 2))
                        * (((-1) + (kernel_size_2 // 2)) <= (1 + (y_coord // 2)))
                        + (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size_2 // 2))))
               < (0 if (0 >= (y_coord // 2)) else (y_coord // 2))))
        ))
        + z_coord * (kernel_size_2 // 2) * (kernel_size_2 // 2)
        + (kernel_size_2 // 2) * (
            (0 if (0 >= (y_coord // 2)) else (y_coord // 2))
            * (
                (0 if (0 >= (y_coord // 2)) else (y_coord // 2)) <=
                ((-1) + (((-1) + (kernel_size_2 // 2))
                         * (((-1) + (kernel_size_2 // 2)) <= (1 + (y_coord // 2)))
                         + (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size_2 // 2))))
            )
            + ((-1) + (((-1) + (kernel_size_2 // 2))
                       * (((-1) + (kernel_size_2 // 2)) <= (1 + (y_coord // 2)))
                       + (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size_2 // 2))))
            * (((-1) + (((-1) + (kernel_size_2 // 2))
                        * (((-1) + (kernel_size_2 // 2)) <= (1 + (y_coord // 2)))
                        + (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size_2 // 2))))
               < (0 if (0 >= (y_coord // 2)) else (y_coord // 2))))
        )
        + ((-2) * z_coord * (kernel_size_2 // 2))
        + (
            (0 if (0 >= (x_coord // 2)) else (x_coord // 2))
            * (
                (0 if (0 >= (x_coord // 2)) else (x_coord // 2)) <=
                ((-1) + (((-1) + (kernel_size_2 // 2))
                         * (((-1) + (kernel_size_2 // 2)) <= (1 + (x_coord // 2)))
                         + (1 + (x_coord // 2)) * ((1 + (x_coord // 2)) < ((-1) + (kernel_size_2 // 2))))
            )
            + ((-1) + (((-1) + (kernel_size_2 // 2))
                       * (((-1) + (kernel_size_2 // 2)) <= (1 + (x_coord // 2)))
                       + (1 + (x_coord // 2)) * ((1 + (x_coord // 2)) < ((-1) + (kernel_size_2 // 2))))
            * (((-1) + (((-1) + (kernel_size_2 // 2))
                        * (((-1) + (kernel_size_2 // 2)) <= (1 + (x_coord // 2)))
                        + (1 + (x_coord // 2)) * ((1 + (x_coord // 2)) < ((-1) + (kernel_size_2 // 2))))
               < (0 if (0 >= (x_coord // 2)) else (x_coord // 2))))
        )
    )
    tmp0 = tl.load(input_ptr + input_index, mask, eviction_policy='evict_last')

    # Load from in_out_ptr
    tmp11 = tl.load(in_out_ptr + linear_index, mask, eviction_policy='evict_last')

    # Calculate average
    tmp1 = tmp0 / 4

    # Calculate masks
    mask_y = (0 if (0 >= (y_coord // 2)) else (y_coord // 2))
    mask_y_limit = ((-1) + (kernel_size_2 // 2)) * (((-1) + (kernel_size_2 // 2)) <= (1 + (y_coord // 2))) + (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size_2 // 2)))
    mask_y_condition = mask_y < mask_y_limit

    mask_x = (0 if (0 >= (x_coord // 2)) else (x_coord // 2))
    mask_x_limit = ((-1) + (kernel_size_2 // 2)) * (((-1) + (kernel_size_2 // 2)) <= (1 + (x_coord // 2))) + (1 + (x_coord // 2)) * ((1 + (x_coord // 2)) < ((-1) + (kernel_size_2 // 2)))
    mask_x_condition = mask_x < mask_x_limit

    combined_mask = mask_y_condition & mask_x_condition

    # Apply mask
    tmp10 = tl.where(combined_mask, tmp1, 0.0)

    # Tanh backward calculation
    tmp13 = tmp11 - 0.5
    tmp14 = tl.extra.cuda.libdevice.tanh(tmp13)
    tmp15 = tmp14 * tmp14
    tmp16 = 1.0 - tmp15
    tmp18 = tmp10 * tmp16

    # Store result
    tl.store(in_out_ptr + linear_index, tmp18, mask)