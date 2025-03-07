# From: 7_GoogleNetInceptionV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_cat_mean_55per_fused_cat_mean_55(
    output_ptr, input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, input_ptr7, 
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 10240
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_index = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_index < rnumel
    x_mod_1024 = x_index % 1024
    r_index_2 = r_index
    x_div_1024 = x_index // 1024
    x_full_index = x_index

    # Load and process input0
    zero_tensor = tl.full([1, 1], 0, tl.int64)
    max_384 = tl.full([1, 1], 384, tl.int64)
    condition_384 = x_mod_1024 < max_384
    load_0 = tl.load(input_ptr0 + (384 * r_index_2 + 18816 * x_div_1024 + x_mod_1024), r_mask & condition_384 & x_mask, eviction_policy='evict_last', other=0.0)
    load_1 = tl.load(input_ptr1 + (tl.broadcast_to(x_mod_1024, [XBLOCK, RBLOCK])), r_mask & condition_384 & x_mask, eviction_policy='evict_last', other=0.0)
    sum_0 = load_0 + load_1
    zero_fill_0 = tl.full(sum_0.shape, 0.0, sum_0.dtype)
    result_0 = tl.where(condition_384, sum_0, zero_fill_0)

    # Load and process input2 and input3
    min_768 = tl.full([1, 1], 768, tl.int64)
    condition_768 = (x_mod_1024 >= max_384) & (x_mod_1024 < min_768)
    load_2 = tl.load(input_ptr2 + (384 * r_index_2 + 18816 * x_div_1024 + (-384 + x_mod_1024)), r_mask & condition_768 & x_mask, eviction_policy='evict_last', other=0.0)
    load_3 = tl.load(input_ptr3 + (tl.broadcast_to(-384 + x_mod_1024, [XBLOCK, RBLOCK])), r_mask & condition_768 & x_mask, eviction_policy='evict_last', other=0.0)
    sum_1 = load_2 + load_3
    zero_fill_1 = tl.full(sum_1.shape, 0.0, sum_1.dtype)
    result_1 = tl.where(condition_768, sum_1, zero_fill_1)

    # Load and process input4 and input5
    max_896 = tl.full([1, 1], 896, tl.int64)
    condition_896 = (x_mod_1024 >= min_768) & (x_mod_1024 < max_896)
    load_4 = tl.load(input_ptr4 + (128 * r_index_2 + 6272 * x_div_1024 + (-768 + x_mod_1024)), r_mask & condition_896 & x_mask, eviction_policy='evict_last', other=0.0)
    load_5 = tl.load(input_ptr5 + (tl.broadcast_to(-768 + x_mod_1024, [XBLOCK, RBLOCK])), r_mask & condition_896 & x_mask, eviction_policy='evict_last', other=0.0)
    sum_2 = load_4 + load_5
    zero_fill_2 = tl.full(sum_2.shape, 0.0, sum_2.dtype)
    result_2 = tl.where(condition_896, sum_2, zero_fill_2)

    # Load and process input6 and input7
    min_1024 = tl.full([1, 1], 1024, tl.int64)
    condition_1024 = x_mod_1024 >= max_896
    load_6 = tl.load(input_ptr6 + (128 * r_index_2 + 6272 * x_div_1024 + (-896 + x_mod_1024)), r_mask & condition_1024 & x_mask, eviction_policy='evict_last', other=0.0)
    load_7 = tl.load(input_ptr7 + (tl.broadcast_to(-896 + x_mod_1024, [XBLOCK, RBLOCK])), r_mask & condition_1024 & x_mask, eviction_policy='evict_last', other=0.0)
    sum_3 = load_6 + load_7
    zero_fill_3 = tl.full(sum_3.shape, 0.0, sum_3.dtype)
    result_3 = tl.where(condition_1024, sum_3, zero_fill_3)

    # Combine results
    combined_result_896 = tl.where(condition_896, result_2, result_3)
    combined_result_768 = tl.where(condition_768, result_1, combined_result_896)
    final_result = tl.where(condition_384, result_0, combined_result_768)

    # Broadcast and sum
    broadcast_result = tl.broadcast_to(final_result, [XBLOCK, RBLOCK])
    masked_result = tl.where(r_mask & x_mask, broadcast_result, 0)
    summed_result = tl.sum(masked_result, 1)[:, None]

    # Calculate mean
    mean_divisor = 49.0
    mean_result = summed_result / mean_divisor

    # Store result
    tl.debug_barrier()
    tl.store(output_ptr + (x_full_index), mean_result, x_mask)