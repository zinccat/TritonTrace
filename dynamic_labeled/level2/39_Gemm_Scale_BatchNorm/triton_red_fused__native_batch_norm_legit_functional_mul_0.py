# From: 39_Gemm_Scale_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_mul_0(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift,
    output_ptr_mean, output_ptr_var, output_ptr_scale, output_ptr_shift,
    kernel_size, num_elements_x, num_elements_r, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
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

    final_mean, final_var, final_weight = triton_helpers.welford(
        running_mean, running_m2, running_weight, 1
    )
    final_mean_broadcasted = final_mean[:, None]
    final_var_broadcasted = final_var[:, None]
    final_weight_broadcasted = final_weight[:, None]

    tl.store(output_ptr_mean + (x_indices_flat), final_mean_broadcasted, x_mask)
    tl.store(output_ptr_var + (x_indices_flat), final_var_broadcasted, x_mask)

    loaded_scale = tl.load(input_ptr_scale + (x_indices_flat), x_mask, eviction_policy='evict_last')
    loaded_shift = tl.load(input_ptr_shift + (x_indices_flat), x_mask, eviction_policy='evict_last')

    kernel_size_float = kernel_size.to(tl.float32)
    normalized_variance = final_var_broadcasted / kernel_size_float
    epsilon = 1e-05
    variance_with_epsilon = normalized_variance + epsilon
    reciprocal_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)

    normalization_factor = (((512 * kernel_size) / 512) / ((tl.full([], -1.00000000000000, tl.float64)) + ((512 * kernel_size) / 512)))
    normalization_factor_float = normalization_factor.to(tl.float32)
    normalized_mean = normalized_variance * normalization_factor_float
    momentum = 0.1
    moving_mean_update = normalized_mean * momentum
    momentum_factor = 0.9
    updated_mean = loaded_scale * momentum_factor + moving_mean_update
    updated_var = final_var_broadcasted * momentum

    tl.store(output_ptr_scale + (x_indices_flat), reciprocal_sqrt_variance, x_mask)
    tl.store(output_ptr_shift + (x_indices_flat), updated_mean, x_mask)
    tl.store(output_ptr_shift + (x_indices_flat), updated_var + (loaded_shift * momentum_factor), x_mask)