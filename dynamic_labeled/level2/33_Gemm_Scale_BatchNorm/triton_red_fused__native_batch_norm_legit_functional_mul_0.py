# From: 33_Gemm_Scale_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_mul_0(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias,
    output_ptr_mean, output_ptr_var, output_ptr_scale, output_ptr_bias,
    output_ptr_running_mean, output_ptr_running_var, kernel_size, num_elements_x, num_elements_r,
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements_x = 512
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    loaded_variances = tl.load(input_ptr_var + (x_indices_flat), x_mask, eviction_policy='evict_last')
    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)

    for r_offset in range(0, num_elements_r, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < num_elements_r
        r_indices_flat = r_indices
        loaded_means = tl.load(input_ptr_mean + (x_indices_flat + 512 * r_indices_flat), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        element_wise_product = loaded_means * loaded_variances
        broadcasted_product = tl.broadcast_to(element_wise_product, [XBLOCK, RBLOCK])
        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_reduce(
            broadcasted_product, running_mean, running_m2, running_weight, r_offset == 0
        )
        running_mean = tl.where(r_mask & x_mask, running_mean_next, running_mean)
        running_m2 = tl.where(r_mask & x_mask, running_m2_next, running_m2)
        running_weight = tl.where(r_mask & x_mask, running_weight_next, running_weight)

    mean, variance, weight = triton_helpers.welford(running_mean, running_m2, running_weight, 1)
    mean_broadcasted = mean[:, None]
    variance_broadcasted = variance[:, None]
    weight_broadcasted = weight[:, None]

    tl.store(output_ptr_mean + (x_indices_flat), mean_broadcasted, x_mask)
    tl.store(output_ptr_var + (x_indices_flat), variance_broadcasted, x_mask)

    loaded_scale = tl.load(input_ptr_scale + (x_indices_flat), x_mask, eviction_policy='evict_last')
    loaded_bias = tl.load(input_ptr_bias + (x_indices_flat), x_mask, eviction_policy='evict_last')
    kernel_size_float = kernel_size.to(tl.float32)
    normalized_variance = variance_broadcasted / kernel_size_float
    epsilon = 1e-05
    variance_with_epsilon = normalized_variance + epsilon
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    scale_factor = (((512 * kernel_size) / 512) / ((tl.full([], -1.00000000000000, tl.float64)) + ((512 * kernel_size) / 512)))
    scale_factor_float = scale_factor.to(tl.float32)
    scaled_variance = normalized_variance * scale_factor_float
    momentum = 0.1
    updated_running_mean = scaled_variance * momentum
    momentum_factor = 0.9
    running_mean_scaled = loaded_scale * momentum_factor
    updated_running_mean += running_mean_scaled
    running_mean_scaled = mean_broadcasted * momentum
    running_var_scaled = loaded_bias * momentum_factor
    updated_running_var = running_mean_scaled + running_var_scaled

    tl.store(output_ptr_scale + (x_indices_flat), reciprocal_sqrt, x_mask)
    tl.store(output_ptr_bias + (x_indices_flat), updated_running_mean, x_mask)
    tl.store(output_ptr_running_mean + (x_indices_flat), updated_running_var, x_mask)
    tl.store(output_ptr_running_var + (x_indices_flat), updated_running_var, x_mask)