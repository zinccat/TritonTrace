# From: 31_Conv2d_Min_Add_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mul_sum_1(in_ptr0, out_ptr0, kernel_size0, kernel_size1, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_num_elements = 240
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_16 = input_index // 16
    input_index_mod_16 = input_index % 16
    temp_result = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_index_3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_2 = reduction_index

        temp_index = reduction_index_2 + input_index_16 * (triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15))
        temp_index_limit = 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1
        index_within_bounds = temp_index < temp_index_limit

        load_index = (
            -2 * (((temp_index // (-2 + kernel_size1)) % (-2 + kernel_size1))) +
            4 * input_index_mod_16 +
            64 * (((temp_index // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0)) +
            kernel_size1 * (((temp_index // (-2 + kernel_size1)) % (-2 + kernel_size1))) +
            input_index_mod_16 * kernel_size1 * kernel_size1 +
            (-64) * kernel_size1 * (((temp_index // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0)) +
            (-4) * kernel_size1 * input_index_mod_16 +
            16 * kernel_size1 * kernel_size1 * (((temp_index // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0)) +
            (temp_index % (-2 + kernel_size1))
        )

        loaded_values = tl.load(in_ptr0 + load_index, reduction_mask & index_within_bounds & input_mask, eviction_policy='evict_last', other=0.0)
        multiplier = 2.0
        multiplied_values = loaded_values * multiplier
        zero_filled_values = tl.full(multiplied_values.shape, 0, multiplied_values.dtype)
        selected_values = tl.where(index_within_bounds, multiplied_values, zero_filled_values)
        broadcasted_values = tl.broadcast_to(selected_values, [XBLOCK, RBLOCK])
        temp_result += broadcasted_values

        temp_result = tl.where(reduction_mask & input_mask, temp_result, temp_result)

    summed_result = tl.sum(temp_result, 1)[:, None]
    tl.store(out_ptr0 + (input_index_3), summed_result, input_mask)