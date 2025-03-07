# From: 41_Gemm_BatchNorm_GELU_GroupNorm_Mean_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_native_group_norm_backward_0(
    input_grad_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, mean_ptr, var_ptr, out_grad_ptr, 
    out_mean_ptr, out_var_ptr, xnumel, rnumel, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_block_index = rindex
    x_block_index = xindex
    x_index_mod_8 = xindex % 8
    x_index_div_8 = xindex // 8

    input_grad = tl.load(input_grad_ptr + (r_block_index + 128 * x_block_index), xmask, other=0.0)
    running_mean = tl.load(running_mean_ptr + (r_block_index + 128 * x_index_mod_8), xmask, eviction_policy='evict_last', other=0.0)
    running_var = tl.load(running_var_ptr + (r_block_index + 128 * x_index_mod_8), xmask, eviction_policy='evict_last', other=0.0)
    weight = tl.load(weight_ptr + (r_block_index + 128 * x_index_mod_8), xmask, eviction_policy='evict_last', other=0.0)
    bias = tl.load(bias_ptr + (r_block_index + 128 * x_index_mod_8), xmask, eviction_policy='evict_last', other=0.0)
    mean = tl.load(mean_ptr + (x_index_div_8), xmask, eviction_policy='evict_last').to(tl.int1)
    var = tl.load(var_ptr + (x_index_div_8), xmask, eviction_policy='evict_last')
    epsilon = tl.load(bias_ptr + (r_block_index + 128 * x_index_mod_8), xmask, eviction_policy='evict_last', other=0.0)

    grad_diff = input_grad - running_mean
    grad_scaled_var = grad_diff * running_var
    grad_scaled_weight = grad_scaled_var * weight
    grad_bias = grad_scaled_weight + bias

    grad_masked = tl.where(mean, 0.0, var)
    inv_std = grad_masked * 0.0009765625
    half_grad_bias = grad_bias * 0.5
    sqrt_2_inv = 0.7071067811865476
    erf_input = grad_bias * sqrt_2_inv
    erf_result = tl.extra.cuda.libdevice.erf(erf_input)
    erf_scaled = (erf_result + 1.0) * half_grad_bias
    scaled_grad = inv_std * erf_scaled
    grad_weighted = scaled_grad * epsilon

    grad_weighted_broadcast = tl.broadcast_to(grad_weighted, [XBLOCK, RBLOCK])
    grad_weighted_masked = tl.where(xmask, grad_weighted_broadcast, 0)
    grad_weighted_sum = tl.sum(grad_weighted_masked, 1)[:, None]

    inv_std_broadcast = tl.broadcast_to(inv_std * epsilon, [XBLOCK, RBLOCK])
    inv_std_masked = tl.where(xmask, inv_std_broadcast, 0)
    inv_std_sum = tl.sum(inv_std_masked, 1)[:, None]

    tl.store(out_grad_ptr + (r_block_index + 128 * x_block_index), grad_bias, xmask)
    tl.store(out_mean_ptr + (x_index_mod_8), grad_weighted_sum, xmask)
    tl.store(out_var_ptr + (x_index_mod_8), inv_std_sum, xmask)