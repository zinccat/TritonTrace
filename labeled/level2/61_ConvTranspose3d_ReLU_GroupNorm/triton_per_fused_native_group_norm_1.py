# From: 61_ConvTranspose3d_ReLU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_native_group_norm_1(input_ptr_mean, input_ptr_var, input_ptr_count, output_ptr_mean, output_ptr_var, output_ptr_count, num_elements, num_groups, XBLOCK: tl.constexpr):
    num_elements = 128
    RBLOCK: tl.constexpr = 4
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    group_indices = r_indices
    element_indices = x_indices
    mean_values = tl.load(input_ptr_mean + (group_indices + (4 * element_indices)), x_mask, other=0.0)
    var_values = tl.load(input_ptr_var + (group_indices + (4 * element_indices)), x_mask, other=0.0)
    count_values = tl.load(input_ptr_count + (group_indices + (4 * element_indices)), x_mask, other=0.0)
    
    broadcast_mean = tl.broadcast_to(mean_values, [XBLOCK, RBLOCK])
    broadcast_var = tl.broadcast_to(var_values, [XBLOCK, RBLOCK])
    broadcast_count = tl.broadcast_to(count_values, [XBLOCK, RBLOCK])
    
    masked_mean = tl.where(x_mask, broadcast_mean, 0)
    masked_var = tl.where(x_mask, broadcast_var, 0)
    masked_count = tl.where(x_mask, broadcast_count, 0)
    
    mean_accum, var_accum, count_accum = triton_helpers.welford(masked_mean, masked_var, masked_count, 1)
    
    reshaped_mean = mean_accum[:, None]
    reshaped_var = var_accum[:, None]
    
    epsilon = 1e-05
    normalized_var = reshaped_var / 51840.0
    adjusted_var = normalized_var + epsilon
    inv_sqrt_var = tl.extra.cuda.libdevice.rsqrt(adjusted_var)
    
    tl.store(output_ptr_count + (element_indices), inv_sqrt_var, x_mask)
    tl.store(output_ptr_mean + (element_indices), reshaped_mean, x_mask)
    tl.store(output_ptr_var + (element_indices), reshaped_var, x_mask)