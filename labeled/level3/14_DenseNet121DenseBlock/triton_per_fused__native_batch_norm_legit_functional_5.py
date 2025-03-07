# From: 14_DenseNet121DenseBlock

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_5(
    input_mean_ptr, input_var_ptr, input_x_ptr, input_beta_ptr, input_gamma_ptr,
    output_mean_ptr, output_var_ptr, output_x_ptr, output_beta_ptr, output_gamma_ptr,
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 64
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < rnumel
    r1 = r_indices
    x0 = x_indices
    loaded_mean = tl.load(input_mean_ptr + (x0 + 64 * r1), r_mask & x_mask, other=0.0)
    loaded_var = tl.load(input_var_ptr + (x0 + 64 * r1), r_mask & x_mask, other=0.0)
    loaded_x = tl.load(input_x_ptr + (x0 + 64 * r1), r_mask & x_mask, other=0.0)
    loaded_beta = tl.load(input_beta_ptr + (x0), x_mask, eviction_policy='evict_last')
    loaded_gamma = tl.load(input_gamma_ptr + (x0), x_mask, eviction_policy='evict_last')
    
    broadcast_mean = tl.broadcast_to(loaded_mean, [XBLOCK, RBLOCK])
    broadcast_var = tl.broadcast_to(loaded_var, [XBLOCK, RBLOCK])
    broadcast_x = tl.broadcast_to(loaded_x, [XBLOCK, RBLOCK])
    
    masked_mean = tl.where(r_mask & x_mask, broadcast_mean, 0)
    masked_var = tl.where(r_mask & x_mask, broadcast_var, 0)
    masked_x = tl.where(r_mask & x_mask, broadcast_x, 0)
    
    mean, var, _ = triton_helpers.welford(masked_mean, masked_var, masked_x, 1)
    reshaped_mean = mean[:, None]
    reshaped_var = var[:, None]
    
    epsilon = 1e-05
    variance_epsilon = reshaped_var / 501760.0
    adjusted_variance = variance_epsilon + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    
    momentum = 0.1
    adjusted_mean = reshaped_mean * momentum
    decay = 0.9
    updated_beta = loaded_beta * decay + adjusted_mean
    updated_gamma = loaded_gamma * decay + reshaped_var * momentum
    
    tl.store(output_mean_ptr + (x0), inv_stddev, x_mask)
    tl.store(output_var_ptr + (x0), updated_beta, x_mask)
    tl.store(output_x_ptr + (x0), updated_gamma, x_mask)
    tl.store(output_beta_ptr + (x0), reshaped_mean, x_mask)
    tl.store(output_gamma_ptr + (x0), reshaped_var, x_mask)