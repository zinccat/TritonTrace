# From: 22_EfficientNetB0

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_mean_relu_71(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 12800
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < rnumel
    r2 = r_indices
    x_col = (x_indices % 1280)
    x_row = x_indices // 1280
    x_flat_index = x_indices
    loaded_data0 = tl.load(in_ptr0 + (x_col + 1280 * r2 + 62720 * x_row), r_mask & x_mask, other=0.0)
    loaded_data1 = tl.load(in_ptr1 + (x_col), x_mask, eviction_policy='evict_last')
    loaded_data2 = tl.load(in_ptr2 + (x_col), x_mask, eviction_policy='evict_last')
    loaded_data3 = tl.load(in_ptr3 + (x_col), x_mask, eviction_policy='evict_last')
    loaded_data4 = tl.load(in_ptr4 + (x_col), x_mask, eviction_policy='evict_last')
    subtracted_data = loaded_data0 - loaded_data1
    multiplied_data1 = subtracted_data * loaded_data2
    multiplied_data2 = multiplied_data1 * loaded_data3
    added_data = multiplied_data2 + loaded_data4
    zero_tensor = tl.full([1, 1], 0, tl.int32)
    max_data = triton_helpers.maximum(zero_tensor, added_data)
    broadcasted_max = tl.broadcast_to(max_data, [XBLOCK, RBLOCK])
    conditional_data = tl.where(r_mask & x_mask, broadcasted_max, 0)
    summed_data = tl.sum(conditional_data, 1)[:, None]
    divisor = 49.0
    result = summed_data / divisor
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x_flat_index), result, x_mask)