# From: 41_Gemm_BatchNorm_GELU_GroupNorm_Mean_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_native_group_norm_1(
    input_ptr_mean, input_ptr_var, input_ptr_beta, input_ptr_gamma, input_ptr_bias,
    output_ptr_mean, output_ptr_var, output_ptr_beta, output_ptr_gamma,
    num_elements, reduction_elements, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 128
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < num_elements
    r_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_index
    x3 = x_index
    x0 = (x_index % 8)
    
    mean = tl.load(input_ptr_mean + (r2 + 128 * x3), x_mask, other=0.0)
    var = tl.load(input_ptr_var + (r2 + 128 * x0), x_mask, eviction_policy='evict_last', other=0.0)
    beta = tl.load(input_ptr_beta + (r2 + 128 * x0), x_mask, eviction_policy='evict_last', other=0.0)
    gamma = tl.load(input_ptr_gamma + (r2 + 128 * x0), x_mask, eviction_policy='evict_last', other=0.0)
    bias = tl.load(input_ptr_bias + (r2 + 128 * x0), x_mask, eviction_policy='evict_last', other=0.0)
    
    normalized = mean - var
    scaled = normalized * beta
    activated = scaled * gamma
    biased = activated + bias
    
    half = 0.5
    sqrt2 = 0.7071067811865476
    erf_input = biased * sqrt2
    erf_result = tl.extra.cuda.libdevice.erf(erf_input)
    one = 1.0
    erf_adjusted = erf_result + one
    gelu_output = half * biased * erf_adjusted
    
    broadcast_gelu = tl.broadcast_to(gelu_output, [XBLOCK, RBLOCK])
    masked_gelu = tl.where(x_mask, broadcast_gelu, 0)
    
    sum_gelu = tl.sum(masked_gelu, 1)[:, None]
    num_elements_float = tl.full([XBLOCK, 1], 128, tl.int32).to(tl.float32)
    mean_gelu = sum_gelu / num_elements_float
    
    centered_gelu = broadcast_gelu - mean_gelu
    squared_gelu = centered_gelu * centered_gelu
    broadcast_squared = tl.broadcast_to(squared_gelu, [XBLOCK, RBLOCK])
    masked_squared = tl.where(x_mask, broadcast_squared, 0)
    
    sum_squared = tl.sum(masked_squared, 1)[:, None]
    variance = sum_squared / 128.0
    epsilon = 1e-05
    variance_adjusted = variance + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_adjusted)
    
    tl.store(output_ptr_mean + (r2 + 128 * x3), biased, x_mask)
    tl.store(output_ptr_gamma + (x3), inv_stddev, x_mask)
    tl.store(output_ptr_beta + (x3), mean_gelu, x_mask)
    tl.store(output_ptr_var + (x3), sum_squared, x_mask)