# From: 41_Gemm_BatchNorm_GELU_GroupNorm_Mean_ReLU

import triton
import triton.language as tl


@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_native_group_norm_1(
    input_mean_ptr, input_var_ptr, input_gamma_ptr, input_beta_ptr, input_x_ptr,
    output_y_ptr, output_mean_ptr, output_var_ptr, output_x_norm_ptr,
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 1024
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_block_index = rindex
    x_block_index = xindex
    x_mod_index = xindex % 8

    mean = tl.load(input_mean_ptr + (r_block_index + (128 * x_block_index)), xmask, other=0.0)
    var = tl.load(input_var_ptr + (r_block_index + (128 * x_mod_index)), xmask, eviction_policy='evict_last', other=0.0)
    gamma = tl.load(input_gamma_ptr + (r_block_index + (128 * x_mod_index)), xmask, eviction_policy='evict_last', other=0.0)
    beta = tl.load(input_beta_ptr + (r_block_index + (128 * x_mod_index)), xmask, eviction_policy='evict_last', other=0.0)
    x = tl.load(input_x_ptr + (r_block_index + (128 * x_mod_index)), xmask, eviction_policy='evict_last', other=0.0)

    x_centered = mean - var
    x_scaled = x_centered * gamma
    x_scaled_var = x_scaled * var
    x_normalized = x_scaled_var + beta

    half = 0.5
    sqrt_2 = 0.7071067811865476
    erf_input = x_normalized * sqrt_2
    erf_result = tl.extra.cuda.libdevice.erf(erf_input)
    one = 1.0
    erf_adjusted = erf_result + one
    x_gelu = x_normalized * half * erf_adjusted

    x_gelu_broadcast = tl.broadcast_to(x_gelu, [XBLOCK, RBLOCK])
    tl.where(xmask, x_gelu_broadcast, 0)

    x_gelu_sum = tl.sum(x_gelu_broadcast, 1)[:, None]
    block_size = tl.full([XBLOCK, 1], 128, tl.int32).to(tl.float32)
    mean_adjusted = x_gelu_sum / block_size
    x_centered_gelu = x_gelu_broadcast - mean_adjusted
    x_centered_squared = x_centered_gelu * x_centered_gelu
    x_centered_squared_broadcast = tl.broadcast_to(x_centered_squared, [XBLOCK, RBLOCK])
    x_centered_squared_sum = tl.sum(tl.where(xmask, x_centered_squared_broadcast, 0), 1)[:, None]
    variance = x_centered_squared_sum / 128.0
    epsilon = 1e-05
    variance_adjusted = variance + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_adjusted)

    tl.store(output_y_ptr + (r_block_index + (128 * x_block_index)), x_normalized, xmask)
    tl.store(output_x_norm_ptr + (x_mod_index), inv_stddev, xmask)
    tl.store(output_mean_ptr + (x_mod_index), mean_adjusted, xmask)
    tl.store(output_var_ptr + (x_mod_index), variance, xmask)