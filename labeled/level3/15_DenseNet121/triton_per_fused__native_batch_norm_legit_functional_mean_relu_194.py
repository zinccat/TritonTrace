# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_mean_relu_194(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 10240
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < rnumel
    r_indices_adjusted = r_indices
    x_indices_adjusted = x_indices
    x_index_mod = x_indices % 1024

    tmp_mean = tl.load(in_ptr0 + (r_indices_adjusted + 49 * x_indices_adjusted), r_mask & x_mask, other=0.0)
    tmp_input = tl.load(in_ptr1 + (x_index_mod), x_mask, eviction_policy='evict_last')
    tmp_variance = tl.load(in_ptr2 + (x_index_mod), x_mask, eviction_policy='evict_last')
    tmp_gamma = tl.load(in_ptr3 + (x_index_mod), x_mask, eviction_policy='evict_last')
    tmp_beta = tl.load(in_ptr4 + (x_index_mod), x_mask, eviction_policy='evict_last')

    tmp_diff = tmp_mean - tmp_input
    tmp_scaled_variance = tmp_diff * tmp_variance
    tmp_scaled_gamma = tmp_scaled_variance * tmp_gamma
    tmp_normalized = tmp_scaled_gamma + tmp_beta

    tmp_zero = tl.full([1, 1], 0, tl.int32)
    tmp_max = triton_helpers.maximum(tmp_zero, tmp_normalized)
    tmp_broadcast = tl.broadcast_to(tmp_max, [XBLOCK, RBLOCK])
    tmp_selected = tl.where(r_mask & x_mask, tmp_broadcast, 0)
    tmp_sum = tl.sum(tmp_selected, 1)[:, None]
    tmp_divisor = 49.0
    tmp_result = tmp_sum / tmp_divisor

    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x_indices_adjusted), tmp_result, x_mask)