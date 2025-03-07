# From: 23_EfficientNetB1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_59(
    input_ptr_mean, input_ptr_var, input_ptr_count, input_ptr_running_mean, input_ptr_running_var,
    output_ptr_inv_std, output_ptr_running_mean, output_ptr_running_var, output_ptr_mean,
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 1152
    rnumel = 5
    RBLOCK: tl.constexpr = 8
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < rnumel
    r1 = r_indices
    x0 = x_indices
    loaded_mean = tl.load(input_ptr_mean + (x0 + 1152 * r1), r_mask & x_mask, other=0.0)
    loaded_var = tl.load(input_ptr_var + (x0 + 1152 * r1), r_mask & x_mask, other=0.0)
    loaded_count = tl.load(input_ptr_count + (x0 + 1152 * r1), r_mask & x_mask, other=0.0)
    running_mean = tl.load(input_ptr_running_mean + (x0), x_mask, eviction_policy='evict_last')
    running_var = tl.load(input_ptr_running_var + (x0), x_mask, eviction_policy='evict_last')
    
    broadcast_mean = tl.broadcast_to(loaded_mean, [XBLOCK, RBLOCK])
    broadcast_var = tl.broadcast_to(loaded_var, [XBLOCK, RBLOCK])
    broadcast_count = tl.broadcast_to(loaded_count, [XBLOCK, RBLOCK])
    
    masked_mean = tl.where(r_mask & x_mask, broadcast_mean, 0)
    masked_var = tl.where(r_mask & x_mask, broadcast_var, 0)
    masked_count = tl.where(r_mask & x_mask, broadcast_count, 0)
    
    mean, var, count = triton_helpers.welford(masked_mean, masked_var, masked_count, 1)
    
    reshaped_mean = mean[:, None]
    reshaped_var = var[:, None]
    
    normalized_var = reshaped_var / 640.0
    epsilon = 1e-05
    adjusted_var = normalized_var + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(adjusted_var)
    
    momentum = 0.1
    updated_mean = reshaped_mean * momentum
    decay = 0.9
    updated_running_mean = updated_mean + running_mean * decay
    
    scale_factor = 1.001564945226917
    scaled_var = normalized_var * scale_factor
    scaled_momentum = scaled_var * momentum
    updated_running_var = scaled_momentum + running_var * decay
    
    tl.store(output_ptr_inv_std + (x0), inv_std, x_mask)
    tl.store(output_ptr_running_mean + (x0), updated_running_mean, x_mask)
    tl.store(output_ptr_running_var + (x0), updated_running_var, x_mask)
    tl.store(output_ptr_mean + (x0), reshaped_mean, x_mask)