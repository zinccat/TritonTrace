# From: 94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_0(input_ptr_mean, input_ptr_var, output_ptr_rsqrt, output_ptr_mean, output_ptr_var, num_elements, num_groups, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 32
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    group_indices = r_indices
    element_indices = x_indices
    element_modulo = (x_indices % 32)
    
    mean_values = tl.load(input_ptr_mean + (group_indices + 32 * element_indices), x_mask, other=0.0)
    var_values = tl.load(input_ptr_var + (group_indices + 32 * element_modulo), x_mask, eviction_policy='evict_last', other=0.0)
    
    sum_values = mean_values + var_values
    neg_one = -1.0
    clamped_values = triton_helpers.maximum(sum_values, neg_one)
    one = 1.0
    min_clamped_values = triton_helpers.minimum(clamped_values, one)
    
    max_value = 20.0
    is_greater_than_max = min_clamped_values > max_value
    exp_values = tl.math.exp(min_clamped_values)
    log1p_values = tl.extra.cuda.libdevice.log1p(exp_values)
    adjusted_values = tl.where(is_greater_than_max, min_clamped_values, log1p_values)
    
    tanh_values = tl.extra.cuda.libdevice.tanh(adjusted_values)
    mish_values = min_clamped_values * tanh_values
    broadcast_mish = tl.broadcast_to(mish_values, [XBLOCK, RBLOCK])
    
    masked_mish = tl.where(x_mask, broadcast_mish, 0)
    sum_mish = tl.sum(masked_mish, 1)[:, None]
    block_size = tl.full([XBLOCK, 1], 32, tl.int32).to(tl.float32)
    mean_mish = sum_mish / block_size
    
    centered_values = broadcast_mish - mean_mish
    squared_values = centered_values * centered_values
    broadcast_squared = tl.broadcast_to(squared_values, [XBLOCK, RBLOCK])
    
    masked_squared = tl.where(x_mask, broadcast_squared, 0)
    sum_squared = tl.sum(masked_squared, 1)[:, None]
    variance = sum_squared / 32.0
    
    epsilon = 1e-05
    adjusted_variance = variance + epsilon
    rsqrt_values = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    
    tl.store(output_ptr_rsqrt + (element_indices), rsqrt_values, x_mask)
    tl.store(output_ptr_mean + (element_indices), mean_mish, x_mask)
    tl.store(output_ptr_var + (element_indices), sum_squared, x_mask)