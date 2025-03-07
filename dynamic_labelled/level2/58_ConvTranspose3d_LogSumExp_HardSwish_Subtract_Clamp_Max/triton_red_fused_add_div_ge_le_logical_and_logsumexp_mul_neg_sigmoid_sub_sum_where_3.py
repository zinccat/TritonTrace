# From: 58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_div_ge_le_logical_and_logsumexp_mul_neg_sigmoid_sub_sum_where_3(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, kernel_size0, kernel_size1, kernel_size2, kernel_size3, 
    x_num_elements, r_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_num_elements = 1968
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_div_16 = x_index // 16
    x_mod_16 = x_index % 16
    temp_result = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x_3d_index = x_index

    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r_2d_index = r_index

        # Calculate temporary indices and masks
        temp_index = r_2d_index + x_div_16 * (
            triton_helpers.div_floor_integer(
                122 + (-1) * kernel_size0 + (-4) * kernel_size0 * kernel_size2 * kernel_size2 + 
                2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
                (-8) * kernel_size0 * kernel_size1 * kernel_size2 + 
                8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
                123
            )
        )
        temp_mask = temp_index < (
            (-1) * kernel_size0 + (-4) * kernel_size0 * kernel_size2 * kernel_size2 + 
            2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
            (-8) * kernel_size0 * kernel_size1 * kernel_size2 + 
            8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2
        )

        # Load data from input_ptr0
        input_data0 = tl.load(
            input_ptr0 + (
                (-1) * (
                    (((r_2d_index + x_div_16 * (
                        triton_helpers.div_floor_integer(
                            122 + (-1) * kernel_size0 + (-4) * kernel_size0 * kernel_size2 * kernel_size2 + 
                            2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
                            (-8) * kernel_size0 * kernel_size1 * kernel_size2 + 
                            8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
                            123
                        )
                    ) // kernel_size3) % kernel_size0)
                ) + (-1) * (
                    (((r_2d_index + x_div_16 * (
                        triton_helpers.div_floor_integer(
                            122 + (-1) * kernel_size0 + (-4) * kernel_size0 * kernel_size2 * kernel_size2 + 
                            2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
                            (-8) * kernel_size0 * kernel_size1 * kernel_size2 + 
                            8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
                            123
                        )
                    ) // ((-1) + 2 * kernel_size2)) % ((-1) + 2 * kernel_size2))
                ) + (-4) * kernel_size2 * (
                    (((r_2d_index + x_div_16 * (
                        triton_helpers.div_floor_integer(
                            122 + (-1) * kernel_size0 + (-4) * kernel_size0 * kernel_size2 * kernel_size2 + 
                            2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
                            (-8) * kernel_size0 * kernel_size1 * kernel_size2 + 
                            8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
                            123
                        )
                    ) // (1 + (-4) * kernel_size2 + 4 * kernel_size2 * kernel_size2)) % ((-1) + 2 * kernel_size1))
                ) + (-4) * kernel_size2 * kernel_size2 * (
                    (((r_2d_index + x_div_16 * (
                        triton_helpers.div_floor_integer(
                            122 + (-1) * kernel_size0 + (-4) * kernel_size0 * kernel_size2 * kernel_size2 + 
                            2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
                            (-8) * kernel_size0 * kernel_size1 * kernel_size2 + 
                            8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
                            123
                        )
                    ) // kernel_size3) % kernel_size0)
                ) + 2 * kernel_size1 * (
                    (((r_2d_index + x_div_16 * (
                        triton_helpers.div_floor_integer(
                            122 + (-1) * kernel_size0 + (-4) * kernel_size0 * kernel_size2 * kernel_size2 + 
                            2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
                            (-8) * kernel_size0 * kernel_size1 * kernel_size2 + 
                            8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
                            123
                        )
                    ) // kernel_size3) % kernel_size0)
                ) + 2 * kernel_size2 * (
                    (((r_2d_index + x_div_16 * (
                        triton_helpers.div_floor_integer(
                            122 + (-1) * kernel_size0 + (-4) * kernel_size0 * kernel_size2 * kernel_size2 + 
                            2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
                            (-8) * kernel_size0 * kernel_size1 * kernel_size2 + 
                            8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
                            123
                        )
                    ) // ((-1) + 2 * kernel_size2)) % ((-1) + 2 * kernel_size2))
                ) + 4 * kernel_size2 * (
                    (((r_2d_index + x_div_16 * (
                        triton_helpers.div_floor_integer(
                            122 + (-1) * kernel_size0 + (-4) * kernel_size0 * kernel_size2 * kernel_size2 + 
                            2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
                            (-8) * kernel_size0 * kernel_size1 * kernel_size2 + 
                            8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
                            123
                        )
                    ) // kernel_size3) % kernel_size0)
                ) + 4 * kernel_size2 * kernel_size2 * (
                    (((r_2d_index + x_div_16 * (
                        triton_helpers.div_floor_integer(
                            122 + (-1) * kernel_size0 + (-4) * kernel_size0 * kernel_size2 * kernel_size2 + 
                            2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
                            (-8) * kernel_size0 * kernel_size1 * kernel_size2 + 
                            8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
                            123
                        )
                    ) // (1 + (-4) * kernel_size2 + 4 * kernel_size2 * kernel_size2)) % ((-1) + 2 * kernel_size1))
                ) + (-8) * kernel_size1 * kernel_size2 * (
                    (((r_2d_index + x_div_16 * (
                        triton_helpers.div_floor_integer(
                            122 + (-1) * kernel_size0 + (-4) * kernel_size0 * kernel_size2 * kernel_size2 + 
                            2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
                            (-8) * kernel_size0 * kernel_size1 * kernel_size2 + 
                            8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
                            123
                        )
                    ) // kernel_size3) % kernel_size0)
                ) + 8 * kernel_size1 * kernel_size2 * kernel_size2 * (
                    (((r_2d_index + x_div_16 * (
                        triton_helpers.div_floor_integer(
                            122 + (-1) * kernel_size0 + (-4) * kernel_size0 * kernel_size2 * kernel_size2 + 
                            2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
                            (-8) * kernel_size0 * kernel_size1 * kernel_size2 + 
                            8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
                            123
                        )
                    ) // kernel_size3) % kernel_size0)
                ) + (((r_2d_index + x_div_16 * (
                    triton_helpers.div_floor_integer(
                        122 + (-1) * kernel_size0 + (-4) * kernel_size0 * kernel_size2 * kernel_size2 + 
                        2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
                        (-8) * kernel_size0 * kernel_size1 * kernel_size2 + 
                        8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
                        123
                    )
                ) % ((-1) + 2 * kernel_size2))) + (
                    (((r_2d_index + x_div_16 * (
                        triton_helpers.div_floor_integer(
                            122 + (-1) * kernel_size0 + (-4) * kernel_size0 * kernel_size2 * kernel_size2 + 
                            2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
                            (-8) * kernel_size0 * kernel_size1 * kernel_size2 + 
                            8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
                            123
                        )
                    ) // (1 + (-4) * kernel_size2 + 4 * kernel_size2 * kernel_size2)) % ((-1) + 2 * kernel_size1))
                )
            ), r_mask & temp_mask & x_mask, eviction_policy='evict_last', other=0.0
        )

        # Apply transformations
        bias = 3.0
        sigmoid_input = input_data0 + bias
        sigmoid_output = tl.sigmoid(sigmoid_input)
        multiplied_data = input_data0 * sigmoid_output
        scaled_data = multiplied_data * 0.16666666666666666

        # Load data from input_ptr1
        input_data1 = tl.load(
            input_ptr1 + tl.broadcast_to(x_mod_16, [XBLOCK, RBLOCK]), 
            r_mask & temp_mask & x_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        # Compute difference and apply conditions
        difference = scaled_data - input_data1
        lower_bound = -1.0
        upper_bound = 1.0
        within_bounds = (difference >= lower_bound) & (difference <= upper_bound)

        # Load data from input_ptr2
        input_data2 = tl.load(
            input_ptr2 + (
                (-1) * x_mod_16 + (-1) * (
                    (((r_2d_index + x_div_16 * (
                        triton_helpers.div_floor_integer(
                            122 + (-1) * kernel_size0 + (-4) * kernel_size0 * kernel_size2 * kernel_size2 + 
                            2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
                            (-8) * kernel_size0 * kernel_size1 * kernel_size2 + 
                            8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
                            123
                        )
                    ) // ((-1) + 2 * kernel_size2)) % ((-1) + 2 * kernel_size2))
                ) + (-16) * (
                    (((r_2d_index + x_div_16 * (
                        triton_helpers.div_floor_integer(
                            122 + (-1) * kernel_size0 + (-4) * kernel_size0 * kernel_size2 * kernel_size2 + 
                            2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
                            (-8) * kernel_size0 * kernel_size1 * kernel_size2 + 
                            8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
                            123
                        )
                    ) // kernel_size3) % kernel_size0)
                ) + (-64) * kernel_size2 * kernel_size2 * (
                    (((r_2d_index + x_div_16 * (
                        triton_helpers.div_floor_integer(
                            122 + (-1) * kernel_size0 + (-4) * kernel_size0 * kernel_size2 * kernel_size2 + 
                            2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
                            (-8) * kernel_size0 * kernel_size1 * kernel_size2 + 
                            8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
                            123
                        )
                    ) // kernel_size3) % kernel_size0)
                ) + (-4) * kernel_size2 * (
                    (((r_2d_index + x_div_16 * (
                        triton_helpers.div_floor_integer(
                            122 + (-1) * kernel_size0 + (-4) * kernel_size0 * kernel_size2 * kernel_size2 + 
                            2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
                            (-8) * kernel_size0 * kernel_size1 * kernel_size2 + 
                            8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
                            123
                        )
                    ) // (1 + (-4) * kernel_size2 + 4 * kernel_size2 * kernel_size2)) % ((-1) + 2 * kernel_size1))
                ) + (-4) * x_mod_16 * kernel_size2 * kernel_size2 + 2 * kernel_size1 * x_mod_16 + 2 * kernel_size2 * (
                    (((r_2d_index + x_div_16 * (
                        triton_helpers.div_floor_integer(
                            122 + (-1) * kernel_size0 + (-4) * kernel_size0 * kernel_size2 * kernel_size2 + 
                            2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
                            (-8) * kernel_size0 * kernel_size1 * kernel_size2 + 
                            8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
                            123
                        )
                    ) // ((-1) + 2 * kernel_size2)) % ((-1) + 2 * kernel_size2))
                ) + 4 * kernel_size2 * x_mod_16 + 4 * kernel_size2 * kernel_size2 * (
                    (((r_2d_index + x_div_16 * (
                        triton_helpers.div_floor_integer(
                            122 + (-1) * kernel_size0 + (-4) * kernel_size0 * kernel_size2 * kernel_size2 + 
                            2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
                            (-8) * kernel_size0 * kernel_size1 * kernel_size2 + 
                            8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
                            123
                        )
                    ) // (1 + (-4) * kernel_size2 + 4 * kernel_size2 * kernel_size2)) % ((-1) + 2 * kernel_size1))
                ) + 32 * kernel_size1 * (
                    (((r_2d_index + x_div_16 * (
                        triton_helpers.div_floor_integer(
                            122 + (-1) * kernel_size0 + (-4) * kernel_size0 * kernel_size2 * kernel_size2 + 
                            2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
                            (-8) * kernel_size0 * kernel_size1 * kernel_size2 + 
                            8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
                            123
                        )
                    ) // kernel_size3) % kernel_size0)
                ) + 64 * kernel_size2 * (
                    (((r_2d_index + x_div_16 * (
                        triton_helpers.div_floor_integer(
                            122 + (-1) * kernel_size0 + (-4) * kernel_size0 * kernel_size2 * kernel_size2 + 
                            2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
                            (-8) * kernel_size0 * kernel_size1 * kernel_size2 + 
                            8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
                            123
                        )
                    ) // kernel_size3) % kernel_size0)
                ) + (-128) * kernel_size1 * kernel_size2 * (
                    (((r_2d_index + x_div_16 * (
                        triton_helpers.div_floor_integer(
                            122 + (-1) * kernel_size0 + (-4) * kernel_size0 * kernel_size2 * kernel_size2 + 
                            2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
                            (-8) * kernel_size0 * kernel_size1 * kernel_size2 + 
                            8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
                            123
                        )
                    ) // kernel_size3) % kernel_size0)
                ) + (-8) * kernel_size1 * kernel_size2 * x_mod_16 + 8 * kernel_size1 * x_mod_16 * kernel_size2 * kernel_size2 + 128 * kernel_size1 * kernel_size2 * kernel_size2 * (
                    (((r_2d_index + x_div_16 * (
                        triton_helpers.div_floor_integer(
                            122 + (-1) * kernel_size0 + (-4) * kernel_size0 * kernel_size2 * kernel_size2 + 
                            2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
                            (-8) * kernel_size0 * kernel_size1 * kernel_size2 + 
                            8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
                            123
                        )
                    ) // kernel_size3) % kernel_size0)
                ) + (((r_2d_index + x_div_16 * (
                    triton_helpers.div_floor_integer(
                        122 + (-1) * kernel_size0 + (-4) * kernel_size0 * kernel_size2 * kernel_size2 + 
                        2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
                        (-8) * kernel_size0 * kernel_size1 * kernel_size2 + 
                        8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
                        123
                    )
                ) % ((-1) + 2 * kernel_size2))) + (
                    (((r_2d_index + x_div_16 * (
                        triton_helpers.div_floor_integer(
                            122 + (-1) * kernel_size0 + (-4) * kernel_size0 * kernel_size2 * kernel_size2 + 
                            2 * kernel_size0 * kernel_size1 + 4 * kernel_size0 * kernel_size2 + 
                            (-8) * kernel_size0 * kernel_size1 * kernel_size2 + 
                            8 * kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2, 
                            123
                        )
                    ) // (1 + (-4) * kernel_size2 + 4 * kernel_size2 * kernel_size2)) % ((-1) + 2 * kernel_size1))
                )
            ), r_mask & temp_mask & x_mask, eviction_policy='evict_last', other=0.0
        )

        # Apply conditional logic
        conditional_result = tl.where(within_bounds, input_data2, 0.0)
        negated_result = -conditional_result
        negated_result_broadcast = tl.where(temp_mask, negated_result, tl.full(negated_result.shape, 0, negated_result.dtype))
        broadcasted_negated_result = tl.broadcast_to(negated_result_broadcast, [XBLOCK, RBLOCK])
        temp_result_update = temp_result + broadcasted_negated_result
        temp_result = tl.where(r_mask & x_mask, temp_result_update, temp_result)

    # Sum and store the result
    summed_result = tl.sum(temp_result, 1)[:, None]
    tl.store(output_ptr0 + (x_3d_index), summed_result, x_mask)