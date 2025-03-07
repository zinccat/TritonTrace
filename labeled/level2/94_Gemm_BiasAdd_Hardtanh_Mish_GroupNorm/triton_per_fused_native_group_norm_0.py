# From: 94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_native_group_norm_0(input_ptr_mean, input_ptr_var, output_ptr_rsqrt_var, output_ptr_mean, output_ptr_var, num_elements, num_groups, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 32
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_indices_2 = r_indices
    x_indices_3 = x_indices
    x_indices_0 = x_indices % 32
    mean_values = tl.load(input_ptr_mean + (r_indices_2 + (32 * x_indices_3)), None)
    var_values = tl.load(input_ptr_var + (r_indices_2 + (32 * x_indices_0)), None, eviction_policy='evict_last')
    sum_mean_var = mean_values + var_values
    neg_one = -1.0
    max_mean_var = triton_helpers.maximum(sum_mean_var, neg_one)
    one = 1.0
    min_mean_var = triton_helpers.minimum(max_mean_var, one)
    threshold = 20.0
    is_greater_than_threshold = min_mean_var > threshold
    exp_min_mean_var = tl.math.exp(min_mean_var)
    log1p_exp_min_mean_var = tl.extra.cuda.libdevice.log1p(exp_min_mean_var)
    tanh_result = tl.where(is_greater_than_threshold, min_mean_var, log1p_exp_min_mean_var)
    tanh_result = tl.extra.cuda.libdevice.tanh(tanh_result)
    product_tanh = min_mean_var * tanh_result
    broadcast_product = tl.broadcast_to(product_tanh, [XBLOCK, RBLOCK])
    sum_broadcast_product = tl.sum(broadcast_product, 1)[:, None]
    block_size = tl.full([XBLOCK, 1], 32, tl.int32)
    block_size_float = block_size.to(tl.float32)
    mean_result = sum_broadcast_product / block_size_float
    diff_mean = broadcast_product - mean_result
    squared_diff = diff_mean * diff_mean
    broadcast_squared_diff = tl.broadcast_to(squared_diff, [XBLOCK, RBLOCK])
    sum_squared_diff = tl.sum(broadcast_squared_diff, 1)[:, None]
    mean_squared_diff = sum_squared_diff / 32.0
    epsilon = 1e-05
    adjusted_var = mean_squared_diff + epsilon
    rsqrt_adjusted_var = tl.extra.cuda.libdevice.rsqrt(adjusted_var)
    tl.store(output_ptr_rsqrt_var + (x_indices_3), rsqrt_adjusted_var, None)
    tl.store(output_ptr_mean + (x_indices_3), mean_result, None)
    tl.store(output_ptr_var + (x_indices_3), sum_squared_diff, None)