# From: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_1(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, kernel_size, 
    input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 384
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_1 = input_index // 64
    input_index_0 = (input_index % 64)
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    variance_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    input_index_3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_2 = reduction_index
        combined_index = reduction_index_2 + input_index_1 * ((5 + 4096 * kernel_size) // 6)
        max_index = 4096 * kernel_size
        valid_mask = combined_index < max_index
        loaded_values = tl.load(
            input_ptr + (4096 * input_index_0 + 262144 * (((combined_index // 4096) % kernel_size)) + (combined_index % 4096)),
            valid_mask & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        zero_values = tl.full(loaded_values.shape, 0, loaded_values.dtype)
        valid_values = tl.where(valid_mask, 0.0, zero_values)
        valid_mask_broadcast = tl.where(valid_mask, 1.0, zero_values)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        broadcasted_valid_values = tl.broadcast_to(valid_values, [XBLOCK, RBLOCK])
        broadcasted_valid_mask = tl.broadcast_to(valid_mask_broadcast, [XBLOCK, RBLOCK])

        mean_accumulator_next, variance_accumulator_next, weight_accumulator_next = triton_helpers.welford_combine(
            mean_accumulator, variance_accumulator, weight_accumulator,
            broadcasted_values, broadcasted_valid_values, broadcasted_valid_mask
        )

        mean_accumulator = tl.where(reduction_mask & input_mask, mean_accumulator_next, mean_accumulator)
        variance_accumulator = tl.where(reduction_mask & input_mask, variance_accumulator_next, variance_accumulator)
        weight_accumulator = tl.where(reduction_mask & input_mask, weight_accumulator_next, weight_accumulator)

    mean_result, variance_result, weight_result = triton_helpers.welford(
        mean_accumulator, variance_accumulator, weight_accumulator, 1
    )

    mean_result = mean_result[:, None]
    variance_result = variance_result[:, None]
    weight_result = weight_result[:, None]

    tl.store(output_mean_ptr + (input_index_3), mean_result, input_mask)
    tl.store(output_var_ptr + (input_index_3), variance_result, input_mask)
    tl.store(output_weight_ptr + (input_index_3), weight_result, input_mask)