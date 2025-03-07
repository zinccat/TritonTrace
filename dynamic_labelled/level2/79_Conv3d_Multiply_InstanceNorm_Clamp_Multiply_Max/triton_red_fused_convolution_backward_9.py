# From: 79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_9(
    input_ptr, output_ptr, kernel_size_d, kernel_size_h, kernel_size_w, kernel_size_c, input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 336
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_channel = input_index % 16
    input_block = input_index // 16
    temp_result = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_linear_index = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_linear_index = reduction_index

        temp_index = reduction_linear_index + input_block * (
            triton_helpers.div_floor_integer(
                20 + (-8 * kernel_size_d) + (-2 * kernel_size_d * kernel_size_w * kernel_size_w) + 
                4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
                kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + 
                (-4 * kernel_size_d * kernel_size_h * kernel_size_w), 
                21
            )
        )

        temp_condition = (
            (-8 * kernel_size_d) + (-2 * kernel_size_d * kernel_size_w * kernel_size_w) + 
            4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
            kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + 
            (-4 * kernel_size_d * kernel_size_h * kernel_size_w)
        )

        condition_mask = temp_index < temp_condition

        input_load_index = (
            (-128) * (
                (((reduction_linear_index + input_block * (
                    triton_helpers.div_floor_integer(
                        20 + (-8 * kernel_size_d) + (-2 * kernel_size_d * kernel_size_w * kernel_size_w) + 
                        4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
                        kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + 
                        (-4 * kernel_size_d * kernel_size_h * kernel_size_w), 
                        21
                    )
                ) // kernel_size_c) % kernel_size_d)
                + (-8) * input_channel
                + (-2) * (
                    ((reduction_linear_index + input_block * (
                        triton_helpers.div_floor_integer(
                            20 + (-8 * kernel_size_d) + (-2 * kernel_size_d * kernel_size_w * kernel_size_w) + 
                            4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
                            kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + 
                            (-4 * kernel_size_d * kernel_size_h * kernel_size_w), 
                            21
                        )
                    ) // (-2 + kernel_size_w)) % (-2 + kernel_size_w))
                )
                + 4 * (
                    (((reduction_linear_index + input_block * (
                        triton_helpers.div_floor_integer(
                            20 + (-8 * kernel_size_d) + (-2 * kernel_size_d * kernel_size_w * kernel_size_w) + 
                            4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
                            kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + 
                            (-4 * kernel_size_d * kernel_size_h * kernel_size_w), 
                            21
                        )
                    ) // (4 + kernel_size_w * kernel_size_w + (-4) * kernel_size_w)) % (-2 + kernel_size_h))
                    + kernel_size_w * (
                        ((reduction_linear_index + input_block * (
                            triton_helpers.div_floor_integer(
                                20 + (-8 * kernel_size_d) + (-2 * kernel_size_d * kernel_size_w * kernel_size_w) + 
                                4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
                                kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + 
                                (-4 * kernel_size_d * kernel_size_h * kernel_size_w), 
                                21
                            )
                        ) // (-2 + kernel_size_w)) % (-2 + kernel_size_w))
                    )
                    + kernel_size_w * kernel_size_w * (
                        (((reduction_linear_index + input_block * (
                            triton_helpers.div_floor_integer(
                                20 + (-8 * kernel_size_d) + (-2 * kernel_size_d * kernel_size_w * kernel_size_w) + 
                                4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
                                kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + 
                                (-4 * kernel_size_d * kernel_size_h * kernel_size_w), 
                                21
                            )
                        ) // (4 + kernel_size_w * kernel_size_w + (-4) * kernel_size_w)) % (-2 + kernel_size_h))
                    )
                    + (-32) * kernel_size_w * kernel_size_w * (
                        ((reduction_linear_index + input_block * (
                            triton_helpers.div_floor_integer(
                                20 + (-8 * kernel_size_d) + (-2 * kernel_size_d * kernel_size_w * kernel_size_w) + 
                                4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
                                kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + 
                                (-4 * kernel_size_d * kernel_size_h * kernel_size_w), 
                                21
                            )
                        ) // kernel_size_c) % kernel_size_d)
                    )
                    + (-4) * kernel_size_w * (
                        (((reduction_linear_index + input_block * (
                            triton_helpers.div_floor_integer(
                                20 + (-8 * kernel_size_d) + (-2 * kernel_size_d * kernel_size_w * kernel_size_w) + 
                                4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
                                kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + 
                                (-4 * kernel_size_d * kernel_size_h * kernel_size_w), 
                                21
                            )
                        ) // (4 + kernel_size_w * kernel_size_w + (-4) * kernel_size_w)) % (-2 + kernel_size_h))
                    )
                    + (-2) * input_channel * kernel_size_w * kernel_size_w
                    + 4 * kernel_h * input_channel
                    + 8 * kernel_w * input_channel
                    + 64 * kernel_h * (
                        ((reduction_linear_index + input_block * (
                            triton_helpers.div_floor_integer(
                                20 + (-8 * kernel_size_d) + (-2 * kernel_size_d * kernel_size_w * kernel_size_w) + 
                                4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
                                kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + 
                                (-4 * kernel_size_d * kernel_size_h * kernel_size_w), 
                                21
                            )
                        ) // kernel_size_c) % kernel_size_d)
                    )
                    + 128 * kernel_w * (
                        ((reduction_linear_index + input_block * (
                            triton_helpers.div_floor_integer(
                                20 + (-8 * kernel_size_d) + (-2 * kernel_size_d * kernel_size_w * kernel_size_w) + 
                                4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
                                kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + 
                                (-4 * kernel_size_d * kernel_size_h * kernel_size_w), 
                                21
                            )
                        ) // kernel_size_c) % kernel_size_d)
                    )
                    + kernel_h * input_channel * kernel_w * kernel_w
                    + (-64) * kernel_h * kernel_w * (
                        ((reduction_linear_index + input_block * (
                            triton_helpers.div_floor_integer(
                                20 + (-8 * kernel_size_d) + (-2 * kernel_size_d * kernel_size_w * kernel_size_w) + 
                                4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
                                kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + 
                                (-4 * kernel_size_d * kernel_size_h * kernel_size_w), 
                                21
                            )
                        ) // kernel_size_c) % kernel_size_d)
                    )
                    + (-4) * kernel_h * kernel_w * input_channel
                    + 16 * kernel_h * kernel_w * kernel_w * (
                        ((reduction_linear_index + input_block * (
                            triton_helpers.div_floor_integer(
                                20 + (-8 * kernel_size_d) + (-2 * kernel_size_d * kernel_size_w * kernel_size_w) + 
                                4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
                                kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + 
                                (-4 * kernel_size_d * kernel_size_h * kernel_size_w), 
                                21
                            )
                        ) // kernel_size_c) % kernel_size_d)
                    )
                    + ((reduction_linear_index + input_block * (
                        triton_helpers.div_floor_integer(
                            20 + (-8 * kernel_size_d) + (-2 * kernel_size_d * kernel_size_w * kernel_size_w) + 
                            4 * kernel_size_d * kernel_size_h + 8 * kernel_size_d * kernel_size_w + 
                            kernel_size_d * kernel_size_h * kernel_size_w * kernel_size_w + 
                            (-4 * kernel_size_d * kernel_size_h * kernel_size_w), 
                            21
                        )
                    )) % (-2 + kernel_w))
                )
            )
        )

        temp_load = tl.load(input_ptr + input_load_index, rmask & condition_mask & input_mask, eviction_policy='evict_last', other=0.0)
        temp_broadcast = tl.broadcast_to(temp_load, [XBLOCK, RBLOCK])
        temp_accumulate = temp_result + temp_broadcast
        temp_result = tl.where(rmask & input_mask, temp_accumulate, temp_result)

    temp_sum = tl.sum(temp_result, 1)[:, None]
    tl.store(output_ptr + input_linear_index, temp_sum, input_mask)