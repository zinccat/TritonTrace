# From: 98_KLDivLoss

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_log_mul_sub_sum_xlogy_0(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xnumel = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = x_index
    _accumulated_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r1 = r_index
        combined_index = r1 + x0 * ((63 + ks0 * ks1) // 64)
        max_index = ks0 * ks1
        index_within_bounds = combined_index < max_index
        value0 = tl.load(in_ptr0 + ((combined_index % max_index)), index_within_bounds & x_mask, eviction_policy='evict_last', other=0.0)
        is_nan0 = tl.extra.cuda.libdevice.isnan(value0).to(tl.int1)
        zero_value = 0.0
        is_zero0 = value0 == zero_value
        log_value0 = tl.math.log(value0)
        xlogy_value0 = value0 * log_value0
        safe_xlogy_value0 = tl.where(is_zero0, zero_value, xlogy_value0)
        nan_replacement = float("nan")
        safe_value0 = tl.where(is_nan0, nan_replacement, safe_xlogy_value0)
        
        value1 = tl.load(in_ptr1 + ((combined_index % max_index)), index_within_bounds & x_mask, eviction_policy='evict_last', other=0.0)
        log_value1 = tl.math.log(value1)
        product_log_value = value0 * log_value1
        difference = safe_value0 - product_log_value
        zero_diff = tl.full(difference.shape, 0, difference.dtype)
        masked_difference = tl.where(index_within_bounds, difference, zero_diff)
        broadcasted_difference = tl.broadcast_to(masked_difference, [XBLOCK, RBLOCK])
        
        accumulated_sum = _accumulated_sum + broadcasted_difference
        _accumulated_sum = tl.where(r_mask & x_mask, accumulated_sum, _accumulated_sum)
    
    final_sum = tl.sum(_accumulated_sum, 1)[:, None]
    tl.store(out_ptr0 + (x0), final_sum, x_mask)