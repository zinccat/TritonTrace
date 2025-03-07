# From: 47_NetVladNoGhostClusters

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional__softmax_3(
    input_ptr_mean, input_ptr_var, input_ptr_gamma, input_ptr_beta, input_ptr_input,
    output_ptr_normalized, output_ptr_softmax, output_ptr_max, output_ptr_sum,
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 3200
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    row_indices = rindex
    col_indices = xindex

    mean = tl.load(input_ptr_mean + (row_indices + 32 * col_indices), xmask, other=0.0)
    variance = tl.load(input_ptr_var + (row_indices), None, eviction_policy='evict_last')
    gamma = tl.load(input_ptr_gamma + (row_indices), None, eviction_policy='evict_last')
    beta = tl.load(input_ptr_beta + (row_indices), None, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (row_indices + 32 * col_indices), xmask, other=0.0)

    normalized_input = input_data - mean
    scaled_variance = normalized_input * variance
    scaled_gamma = scaled_variance * gamma
    shifted_beta = scaled_gamma + beta

    broadcast_shifted_beta = tl.broadcast_to(shifted_beta, [XBLOCK, RBLOCK])
    masked_shifted_beta = tl.where(xmask, broadcast_shifted_beta, float("-inf"))
    max_values = triton_helpers.max2(masked_shifted_beta, 1)[:, None]
    shifted_values = shifted_beta - max_values
    exp_values = tl.math.exp(shifted_values)
    broadcast_exp_values = tl.broadcast_to(exp_values, [XBLOCK, RBLOCK])
    masked_exp_values = tl.where(xmask, broadcast_exp_values, 0)
    sum_exp_values = tl.sum(masked_exp_values, 1)[:, None]
    softmax_values = exp_values / sum_exp_values

    tl.store(output_ptr_normalized + (row_indices + 32 * col_indices), shifted_beta, xmask)
    tl.store(output_ptr_softmax + (row_indices + 32 * col_indices), softmax_values, xmask)
    tl.store(output_ptr_max + (col_indices), max_values, xmask)
    tl.store(output_ptr_sum + (col_indices), sum_exp_values, xmask)