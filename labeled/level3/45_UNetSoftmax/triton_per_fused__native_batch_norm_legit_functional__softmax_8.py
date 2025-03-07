# From: 45_UNetSoftmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional__softmax_8per_fused__native_batch_norm_legit_functional__softmax_8(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel):

    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 256

    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    r_index = tl.arange(0, RBLOCK)[:]

    input_value = tl.load(in_ptr0 + (r_index + 256 * x_index), None)
    mean_value = tl.load(in_ptr1 + (x_index // 32 % 128), None, eviction_policy='evict_last')
    variance_value = tl.load(in_ptr2 + (x_index // 32 % 128), None, eviction_policy='evict_last')
    gamma_value = tl.load(in_ptr3 + (x_index // 32 % 128), None, eviction_policy='evict_last')
    beta_value = tl.load(in_ptr4 + (x_index // 32 % 128), None, eviction_policy='evict_last')

    normalized_value = input_value - mean_value
    variance_scaled = variance_value / 65536.0
    epsilon = 1e-05
    variance_adjusted = variance_scaled + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_adjusted)
    scaled_value = normalized_value * inv_stddev
    gamma_scaled = scaled_value * gamma_value
    beta_shifted = gamma_scaled + beta_value

    exp_values = tl.math.exp(beta_shifted - tl.broadcast_to(tl.max(beta_shifted, 0), [RBLOCK]))
    sum_exp_values = tl.sum(exp_values, 0)
    softmax_output = exp_values / tl.broadcast_to(sum_exp_values, [RBLOCK])

    tl.store(in_out_ptr0 + (r_index + 256 * x_index), softmax_output, None)