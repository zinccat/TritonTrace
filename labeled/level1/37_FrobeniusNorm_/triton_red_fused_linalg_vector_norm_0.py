# From: 37_FrobeniusNorm_

import triton
import triton.language as tl


@triton.jit
def triton_red_fused_linalg_vector_norm_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xnumel = 328
    rnumel = 204601
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
        temp_indices = r_indices_flat + (204601 * x_indices_flat)
        max_index = tl.full([1, 1], 67108864, tl.int32)
        valid_indices = temp_indices < max_index
        loaded_values = tl.load(in_ptr0 + ((r_indices_flat + (204601 * x_indices_flat)) % 67108864), r_mask & valid_indices & x_mask, eviction_policy='evict_last', other=0.0)
        squared_values = loaded_values * loaded_values
        zero_values = tl.full(squared_values.shape, 0, squared_values.dtype)
        masked_squared_values = tl.where(valid_indices, squared_values, zero_values)
        broadcasted_values = tl.broadcast_to(masked_squared_values, [XBLOCK, RBLOCK])
        temp_sum += broadcasted_values
        temp_sum = tl.where(r_mask & x_mask, temp_sum, temp_sum)
    
    sum_result = tl.sum(temp_sum, 1)[:, None]
    tl.store(out_ptr0 + (x_indices_flat), sum_result, x_mask)