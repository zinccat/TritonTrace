# From: 46_NetVladWithGhostClusters

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional__softmax_3(
    input_ptr_mean, input_ptr_var, input_ptr_beta, input_ptr_gamma, input_ptr_input,
    output_ptr_normalized, output_ptr_softmax, output_ptr_max, output_ptr_sum,
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 3200
    rnumel = 48
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < rnumel
    r_indices_adjusted = r_indices
    x_indices_adjusted = x_indices

    mean = tl.load(input_ptr_mean + (r_indices_adjusted + 48 * x_indices_adjusted), r_mask & x_mask, other=0.0)
    variance = tl.load(input_ptr_var + (r_indices_adjusted), r_mask, eviction_policy='evict_last', other=0.0)
    beta = tl.load(input_ptr_beta + (r_indices_adjusted), r_mask, eviction_policy='evict_last', other=0.0)
    gamma = tl.load(input_ptr_gamma + (r_indices_adjusted), r_mask, eviction_policy='evict_last', other=0.0)
    input_data = tl.load(input_ptr_input + (r_indices_adjusted), r_mask, eviction_policy='evict_last', other=0.0)

    normalized_input = input_data - mean
    scaled_variance = normalized_input * variance
    scaled_input = scaled_variance * gamma
    shifted_input = scaled_input + beta

    broadcast_shifted_input = tl.broadcast_to(shifted_input, [XBLOCK, RBLOCK])
    masked_shifted_input = tl.where(r_mask & x_mask, broadcast_shifted_input, float("-inf"))
    max_values = triton_helpers.max2(masked_shifted_input, 1)[:, None]
    shifted_input_centered = shifted_input - max_values

    exp_values = tl.math.exp(shifted_input_centered)
    broadcast_exp_values = tl.broadcast_to(exp_values, [XBLOCK, RBLOCK])
    masked_exp_values = tl.where(r_mask & x_mask, broadcast_exp_values, 0)
    sum_exp_values = tl.sum(masked_exp_values, 1)[:, None]
    softmax_values = exp_values / sum_exp_values

    tl.store(output_ptr_normalized + (r_indices_adjusted + 48 * x_indices_adjusted), shifted_input, r_mask & x_mask)
    tl.store(output_ptr_softmax + (r_indices_adjusted + 48 * x_indices_adjusted), softmax_values, r_mask & x_mask)
    tl.store(output_ptr_max + (x_indices_adjusted), max_values, x_mask)
    tl.store(output_ptr_sum + (x_indices_adjusted), sum_exp_values, x_mask)