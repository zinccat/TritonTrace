# From: 37_FrobeniusNorm_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_linalg_vector_norm_0(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xnumel = 328
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r_indices_flat = r_indices
        temp_indices = r_indices_flat + x_indices_flat * ((327 + ks0 * ks1 * ks2 * ks2) // 328)
        max_index = ks0 * ks1 * ks2 * ks2
        valid_mask = temp_indices < max_index
        loaded_values = tl.load(
            in_ptr0 + ((temp_indices % max_index)),
            valid_mask & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        squared_values = loaded_values * loaded_values
        zero_filled = tl.full(squared_values.shape, 0, squared_values.dtype)
        masked_squared_values = tl.where(valid_mask, squared_values, zero_filled)
        broadcasted_values = tl.broadcast_to(masked_squared_values, [XBLOCK, RBLOCK])
        temp_sum += broadcasted_values
        temp_sum = tl.where(r_mask & x_mask, temp_sum, temp_sum)
    
    summed_values = tl.sum(temp_sum, 1)[:, None]
    tl.store(out_ptr0 + (x_indices_flat), summed_values, x_mask)