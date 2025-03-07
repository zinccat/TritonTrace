# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_mean_relu_314(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 19200
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < rnumel
    r_block_indices = r_indices
    x_block_indices = x_indices
    x_channel_indices = x_indices % 1920

    tmp_mean = tl.load(in_ptr0 + (r_block_indices + 49 * x_block_indices), r_mask & x_mask, other=0.0)
    tmp_input = tl.load(in_ptr1 + (x_channel_indices), x_mask, eviction_policy='evict_last')
    tmp_var = tl.load(in_ptr2 + (x_channel_indices), x_mask, eviction_policy='evict_last')
    tmp_gamma = tl.load(in_ptr3 + (x_channel_indices), x_mask, eviction_policy='evict_last')
    tmp_beta = tl.load(in_ptr4 + (x_channel_indices), x_mask, eviction_policy='evict_last')

    tmp_diff = tmp_mean - tmp_input
    tmp_scaled_var = tmp_diff * tmp_var
    tmp_scaled_gamma = tmp_scaled_var * tmp_gamma
    tmp_normalized = tmp_scaled_gamma + tmp_beta

    tmp_zero = tl.full([1, 1], 0, tl.int32)
    tmp_relu = triton_helpers.maximum(tmp_zero, tmp_normalized)
    tmp_broadcast_relu = tl.broadcast_to(tmp_relu, [XBLOCK, RBLOCK])
    tmp_masked_relu = tl.where(r_mask & x_mask, tmp_broadcast_relu, 0)
    tmp_sum_relu = tl.sum(tmp_masked_relu, 1)[:, None]
    tmp_num_elements = 49.0
    tmp_mean_relu = tmp_sum_relu / tmp_num_elements

    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x_block_indices), tmp_mean_relu, x_mask)