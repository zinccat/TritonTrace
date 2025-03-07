# From: 19_MobileNetV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_28red_fused__native_batch_norm_legit_functional_28(
    input_ptr, output_mean_ptr, output_var_ptr, output_count_ptr, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    rnumel = 123
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_base = tl.arange(0, RBLOCK)[None, :]
    x1 = x_index // 512
    x0 = (x_index % 512)
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    m2_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    count_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = x_index

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r2 = r_index
        tmp_index = r2 + 123 * x1
        max_index = tl.full([1, 1], 1960, tl.int32)
        index_mask = tmp_index < max_index
        loaded_values = tl.load(input_ptr + (x0 + 512 * ((r2 + 123 * x1) % 1960)), r_mask & index_mask, eviction_policy='evict_first', other=0.0)
        zero_values = tl.full(loaded_values.shape, 0, loaded_values.dtype)
        mask_values = tl.where(index_mask, 0.0, zero_values)
        one_values = tl.full(loaded_values.shape, 0, 1.0)
        broadcast_mask = tl.where(index_mask, 1.0, one_values)
        
        broadcast_loaded = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        broadcast_mask_values = tl.broadcast_to(mask_values, [XBLOCK, RBLOCK])
        broadcast_broadcast_mask = tl.broadcast_to(broadcast_mask, [XBLOCK, RBLOCK])
        
        mean_next, m2_next, count_next = triton_helpers.welford_combine(
            mean_accumulator, m2_accumulator, count_accumulator,
            broadcast_loaded, broadcast_mask_values, broadcast_broadcast_mask
        )
        
        mean_accumulator = tl.where(r_mask, mean_next, mean_accumulator)
        m2_accumulator = tl.where(r_mask, m2_next, m2_accumulator)
        count_accumulator = tl.where(r_mask, count_next, count_accumulator)

    mean_result, variance_result, count_result = triton_helpers.welford(
        mean_accumulator, m2_accumulator, count_accumulator, 1
    )
    
    mean_result = mean_result[:, None]
    variance_result = variance_result[:, None]
    count_result = count_result[:, None]
    
    tl.store(output_mean_ptr + (x3), mean_result, None)
    tl.store(output_var_ptr + (x3), variance_result, None)
    tl.store(output_count_ptr + (x3), count_result, None)