# From: 45_UNetSoftmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional__softmax_3per_fused__native_batch_norm_legit_functional__softmax_3(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel):

    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 512

    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[:]
    tl.full([RBLOCK], True, tl.int1)
    r3 = r_index
    x4 = x_index
    x1 = ((x_index // 64) % 64)

    input_value = tl.load(in_ptr0 + (r3 + 512 * x4), None)
    mean_value = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    variance_value = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    gamma_value = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    beta_value = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')

    normalized_value = input_value - mean_value
    variance_scale = 262144.0
    epsilon = 1e-05

    adjusted_variance = variance_value / variance_scale
    variance_with_epsilon = adjusted_variance + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)

    scaled_value = normalized_value * inv_sqrt_variance
    scaled_gamma = scaled_value * gamma_value
    shifted_value = scaled_gamma + beta_value

    broadcast_shifted_value = tl.broadcast_to(shifted_value, [RBLOCK])
    max_value = triton_helpers.promote_to_tensor(triton_helpers.max2(broadcast_shifted_value, 0))
    shifted_for_exp = shifted_value - max_value

    exp_values = tl.math.exp(shifted_for_exp)
    broadcast_exp_values = tl.broadcast_to(exp_values, [RBLOCK])
    sum_exp_values = triton_helpers.promote_to_tensor(tl.sum(broadcast_exp_values, 0))

    softmax_output = exp_values / sum_exp_values
    tl.store(in_out_ptr0 + (r3 + 512 * x4), softmax_output, None)