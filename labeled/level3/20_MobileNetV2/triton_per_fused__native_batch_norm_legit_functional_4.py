# From: 20_MobileNetV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_4(
    input_ptr_mean, input_ptr_var, input_ptr_gamma, input_ptr_beta, input_ptr_input, 
    output_ptr_rsqrt, output_ptr_running_mean, output_ptr_running_var, output_ptr_output, 
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 32
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex

    # Load input data
    input_data = tl.load(input_ptr_input + (x0 + 32 * r1), xmask, other=0.0)
    gamma_data = tl.load(input_ptr_gamma + (x0 + 32 * r1), xmask, other=0.0)
    beta_data = tl.load(input_ptr_beta + (x0 + 32 * r1), xmask, other=0.0)
    running_mean_data = tl.load(input_ptr_mean + (x0), xmask, eviction_policy='evict_last')
    running_var_data = tl.load(input_ptr_var + (x0), xmask, eviction_policy='evict_last')

    # Broadcast to match dimensions
    input_broadcast = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
    gamma_broadcast = tl.broadcast_to(gamma_data, [XBLOCK, RBLOCK])
    beta_broadcast = tl.broadcast_to(beta_data, [XBLOCK, RBLOCK])

    # Apply mask
    masked_input = tl.where(xmask, input_broadcast, 0)
    masked_gamma = tl.where(xmask, gamma_broadcast, 0)
    masked_beta = tl.where(xmask, beta_broadcast, 0)

    # Compute Welford's algorithm for mean and variance
    mean, var, _ = triton_helpers.welford(masked_input, masked_gamma, masked_beta, 1)

    # Reshape mean and variance
    mean_reshaped = mean[:, None]
    var_reshaped = var[:, None]

    # Compute rsqrt of variance
    epsilon = 1e-05
    variance_normalized = var_reshaped / 125440.0
    variance_with_epsilon = variance_normalized + epsilon
    rsqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)

    # Update running mean and variance
    momentum = 0.1
    updated_running_mean = mean_reshaped * momentum + running_mean_data * 0.9
    variance_scale = 1.0000079720023278
    scaled_variance = variance_normalized * variance_scale
    updated_running_var = scaled_variance * momentum + running_var_data * 0.9

    # Store results
    tl.store(output_ptr_rsqrt + (x0), rsqrt_variance, xmask)
    tl.store(output_ptr_running_mean + (x0), updated_running_mean, xmask)
    tl.store(output_ptr_running_var + (x0), updated_running_var, xmask)
    tl.store(output_ptr_output + (x0), mean_reshaped, xmask)