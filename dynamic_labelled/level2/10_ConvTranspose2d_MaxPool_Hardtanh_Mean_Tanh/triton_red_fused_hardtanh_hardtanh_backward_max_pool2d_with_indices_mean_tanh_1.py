# From: 10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_hardtanh_hardtanh_backward_max_pool2d_with_indices_mean_tanh_1(
    in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, kernel_size, input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_indices = input_index

    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_col = reduction_index % kernel_size
        reduction_index_row = reduction_index // kernel_size
        reduction_index_flat = reduction_index

        input_value_0 = tl.load(
            in_ptr0 + (2 * reduction_index_col + 4 * kernel_size * reduction_index_row + 4 * input_indices * kernel_size * kernel_size),
            reduction_mask & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        input_value_1 = tl.load(
            in_ptr0 + (1 + 2 * reduction_index_col + 4 * kernel_size * reduction_index_row + 4 * input_indices * kernel_size * kernel_size),
            reduction_mask & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        input_value_2 = tl.load(
            in_ptr0 + (2 * kernel_size + 2 * reduction_index_col + 4 * kernel_size * reduction_index_row + 4 * input_indices * kernel_size * kernel_size),
            reduction_mask & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        input_value_3 = tl.load(
            in_ptr0 + (1 + 2 * kernel_size + 2 * reduction_index_col + 4 * kernel_size * reduction_index_row + 4 * input_indices * kernel_size * kernel_size),
            reduction_mask & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        is_greater_1 = input_value_1 > input_value_0
        mask_1 = tl.full([1, 1], 1, tl.int8)
        mask_0 = tl.full([1, 1], 0, tl.int8)
        max_index_1 = tl.where(is_greater_1, mask_1, mask_0)

        max_value_1 = triton_helpers.maximum(input_value_1, input_value_0)
        is_greater_2 = input_value_2 > max_value_1
        mask_2 = tl.full([1, 1], 2, tl.int8)
        max_index_2 = tl.where(is_greater_2, mask_2, max_index_1)

        max_value_2 = triton_helpers.maximum(input_value_2, max_value_1)
        is_greater_3 = input_value_3 > max_value_2
        mask_3 = tl.full([1, 1], 3, tl.int8)
        max_index_3 = tl.where(is_greater_3, mask_3, max_index_2)

        max_value_3 = triton_helpers.maximum(input_value_3, max_value_2)

        lower_bound = -1.0
        upper_bound = 1.0
        is_out_of_bounds = (max_value_3 <= lower_bound) | (max_value_3 >= upper_bound)

        clamped_value = triton_helpers.maximum(max_value_3, lower_bound)
        clamped_value = triton_helpers.minimum(clamped_value, upper_bound)

        broadcast_clamped_value = tl.broadcast_to(clamped_value, [XBLOCK, RBLOCK])
        temp_accumulator = temp_accumulator + broadcast_clamped_value
        temp_accumulator = tl.where(reduction_mask & input_mask, temp_accumulator, temp_accumulator)

        tl.store(out_ptr0 + (reduction_index_flat + input_indices * kernel_size * kernel_size), max_index_3, reduction_mask & input_mask)
        tl.store(out_ptr1 + (reduction_index_flat + input_indices * kernel_size * kernel_size), is_out_of_bounds, reduction_mask & input_mask)

    sum_accumulator = tl.sum(temp_accumulator, 1)[:, None]
    kernel_area = kernel_size * kernel_size
    kernel_area_float = kernel_area.to(tl.float32)
    mean_value = sum_accumulator / kernel_area_float
    tanh_value = tl.extra.cuda.libdevice.tanh(mean_value)

    tl.debug_barrier()
    tl.store(in_out_ptr0 + (input_indices), tanh_value, input_mask)