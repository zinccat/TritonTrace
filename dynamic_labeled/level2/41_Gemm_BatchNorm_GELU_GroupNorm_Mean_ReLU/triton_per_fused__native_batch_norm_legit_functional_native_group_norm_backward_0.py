# From: 41_Gemm_BatchNorm_GELU_GroupNorm_Mean_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_native_group_norm_backward_0(
    input_grad_ptr, mean_ptr, variance_ptr, weight_ptr, running_mean_ptr, running_var_ptr, input_ptr, output_grad_ptr, 
    output_mean_ptr, output_var_ptr, input_num_elements, running_num_elements, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 128
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < input_num_elements
    r_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_index
    x3 = x_index
    x0 = (x_index % 8)
    x1 = x_index // 8

    input_grad = tl.load(input_grad_ptr + (r2 + 128 * x3), x_mask, other=0.0)
    mean = tl.load(mean_ptr + (r2 + 128 * x0), x_mask, eviction_policy='evict_last', other=0.0)
    variance = tl.load(variance_ptr + (r2 + 128 * x0), x_mask, eviction_policy='evict_last', other=0.0)
    weight = tl.load(weight_ptr + (r2 + 128 * x0), x_mask, eviction_policy='evict_last', other=0.0)
    running_mean = tl.load(running_mean_ptr + (r2 + 128 * x0), x_mask, eviction_policy='evict_last', other=0.0)
    running_var = tl.load(running_var_ptr + (r2 + 128 * x0), x_mask, eviction_policy='evict_last', other=0.0)
    input_mask = tl.load(input_ptr + (x1), x_mask, eviction_policy='evict_last').to(tl.int1)
    input_data = tl.load(input_ptr + (x1), x_mask, eviction_policy='evict_last')

    normalized_input = input_grad - mean
    scaled_variance = normalized_input * variance
    weighted_input = scaled_variance * weight
    adjusted_input = weighted_input + running_mean
    scale_factor = 0.0
    adjusted_scale = tl.where(input_mask, scale_factor, input_data)
    scale_multiplier = 0.0009765625
    scaled_adjusted = adjusted_scale * scale_multiplier
    half = 0.5
    half_adjusted_input = adjusted_input * half
    sqrt_2 = 0.7071067811865476
    sqrt_2_adjusted_input = adjusted_input * sqrt_2
    erf_result = tl.extra.cuda.libdevice.erf(sqrt_2_adjusted_input)
    one = 1.0
    erf_plus_one = erf_result + one
    scaled_erf = half_adjusted_input * erf_plus_one
    final_scale = scaled_adjusted * scaled_erf
    final_result = final_scale * running_var

    broadcasted_result = tl.broadcast_to(final_result, [XBLOCK, RBLOCK])
    masked_result = tl.where(x_mask, broadcasted_result, 0)
    summed_result = tl.sum(masked_result, 1)[:, None]

    scaled_running_var = scaled_adjusted * running_var
    broadcasted_scaled_var = tl.broadcast_to(scaled_running_var, [XBLOCK, RBLOCK])
    masked_scaled_var = tl.where(x_mask, broadcasted_scaled_var, 0)
    summed_scaled_var = tl.sum(masked_scaled_var, 1)[:, None]

    tl.store(output_grad_ptr + (r2 + 128 * x3), adjusted_input, x_mask)
    tl.store(output_mean_ptr + (x3), summed_result, x_mask)
    tl.store(output_var_ptr + (x3), summed_scaled_var, x_mask)