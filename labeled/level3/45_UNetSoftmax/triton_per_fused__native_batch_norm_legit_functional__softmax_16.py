# From: 45_UNetSoftmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional__softmax_16per_fused__native_batch_norm_legit_functional__softmax_16(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = r_index
    x4 = x_index
    x1 = ((x_index // 8) % 512)
    
    mean = tl.load(in_ptr0 + (r3 + 64 * x4), None)
    variance = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    variance_sqrt = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    gamma = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    beta = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    
    normalized = mean - variance
    variance_scale = 4096.0
    epsilon = 1e-05
    variance_adjusted = variance_sqrt / variance_scale
    variance_adjusted_epsilon = variance_adjusted + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_adjusted_epsilon)
    scaled_normalized = normalized * inv_sqrt_variance
    scaled_gamma = scaled_normalized * gamma
    shifted = scaled_gamma + beta
    
    broadcast_shifted = tl.broadcast_to(shifted, [XBLOCK, RBLOCK])
    max_shifted = triton_helpers.max2(broadcast_shifted, 1)[:, None]
    shifted_exp = shifted - max_shifted
    exp_shifted = tl.math.exp(shifted_exp)
    broadcast_exp = tl.broadcast_to(exp_shifted, [XBLOCK, RBLOCK])
    sum_exp = tl.sum(broadcast_exp, 1)[:, None]
    softmax_output = exp_shifted / sum_exp
    
    tl.store(in_out_ptr0 + (r3 + 64 * x4), softmax_output, None)