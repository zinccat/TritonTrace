# From: 94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_0per_fused_native_group_norm_0(
    input_ptr_mean, input_ptr_var, output_ptr_rsqrt_var, output_ptr_mean, output_ptr_var, 
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_block_index = rindex
    x_block_index = xindex
    x_block_modulo = xindex % 32
    mean_accumulator = tl.load(input_ptr_mean + (r_block_index + 32 * x_block_index), xmask, other=0.0)
    var_accumulator = tl.load(input_ptr_var + (r_block_index + 32 * x_block_modulo), xmask, eviction_policy='evict_last', other=0.0)
    sum_mean_var = mean_accumulator + var_accumulator
    neg_one = -1.0
    clamped_sum = triton_helpers.maximum(sum_mean_var, neg_one)
    one = 1.0
    min_clamped_sum = triton_helpers.minimum(clamped_sum, one)
    max_value = 20.0
    is_greater_than_max = min_clamped_sum > max_value
    exp_clamped_sum = tl.math.exp(min_clamped_sum)
    log1p_exp = tl.extra.cuda.libdevice.log1p(exp_clamped_sum)
    mish_activation = tl.where(is_greater_than_max, min_clamped_sum, log1p_exp)
    tanh_activation = tl.extra.cuda.libdevice.tanh(mish_activation)
    mish_tanh_product = min_clamped_sum * tanh_activation
    broadcast_mish_tanh = tl.broadcast_to(mish_tanh_product, [XBLOCK, RBLOCK])
    masked_mish_tanh = tl.where(xmask, broadcast_mish_tanh, 0)
    sum_masked_mish_tanh = tl.sum(masked_mish_tanh, 1)[:, None]
    block_size = tl.full([XBLOCK, 1], 32, tl.int32)
    block_size_float = block_size.to(tl.float32)
    mean = sum_masked_mish_tanh / block_size_float
    centered_mish_tanh = mish_tanh_product - mean
    squared_centered = centered_mish_tanh * centered_mish_tanh
    broadcast_squared = tl.broadcast_to(squared_centered, [XBLOCK, RBLOCK])
    masked_squared = tl.where(xmask, broadcast_squared, 0)
    sum_masked_squared = tl.sum(masked_squared, 1)[:, None]
    variance = sum_masked_squared / 32.0
    epsilon = 1e-05
    variance_with_epsilon = variance + epsilon
    rsqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    tl.store(output_ptr_rsqrt_var + (x_block_index), rsqrt_variance, xmask)
    tl.store(output_ptr_mean + (x_block_index), mean, xmask)
    tl.store(output_ptr_var + (x_block_index), variance, xmask)