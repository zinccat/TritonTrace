# From: 21_EfficientNetMBConv

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_10(
    input_mean_ptr, input_var_ptr, input_count_ptr, running_mean_ptr, running_var_ptr,
    output_mean_ptr, output_var_ptr, output_running_mean_ptr, output_running_var_ptr, output_count_ptr,
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 192
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < rnumel
    r1 = r_indices
    x0 = x_indices
    loaded_mean = tl.load(input_mean_ptr + (x0 + 192 * r1), r_mask & x_mask, other=0.0)
    loaded_var = tl.load(input_var_ptr + (x0 + 192 * r1), r_mask & x_mask, other=0.0)
    loaded_count = tl.load(input_count_ptr + (x0 + 192 * r1), r_mask & x_mask, other=0.0)
    running_mean = tl.load(running_mean_ptr + (x0), x_mask, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (x0), x_mask, eviction_policy='evict_last')
    
    broadcast_mean = tl.broadcast_to(loaded_mean, [XBLOCK, RBLOCK])
    broadcast_var = tl.broadcast_to(loaded_var, [XBLOCK, RBLOCK])
    broadcast_count = tl.broadcast_to(loaded_count, [XBLOCK, RBLOCK])
    
    masked_mean = tl.where(r_mask & x_mask, broadcast_mean, 0)
    masked_var = tl.where(r_mask & x_mask, broadcast_var, 0)
    masked_count = tl.where(r_mask & x_mask, broadcast_count, 0)
    
    mean, var, count = triton_helpers.welford(masked_mean, masked_var, masked_count, 1)
    reshaped_mean = mean[:, None]
    reshaped_var = var[:, None]
    
    epsilon = 1e-05
    normalized_var = reshited_var / 125440.0
    adjusted_var = normalized_var + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(adjusted_var)
    
    momentum = 0.1
    adjusted_mean = normalized_var * momentum
    decay = 0.9
    updated_running_mean = running_mean * decay + adjusted_mean
    updated_running_var = reshaped_mean * momentum + running_var * decay
    
    tl.store(output_var_ptr + (x0), inv_std, x_mask)
    tl.store(output_running_mean_ptr + (x0), updated_running_mean, x_mask)
    tl.store(output_running_var_ptr + (x0), updated_running_var, x_mask)
    tl.store(output_mean_ptr + (x0), reshaped_mean, x_mask)
    tl.store(output_count_ptr + (x0), reshaped_var, x_mask)