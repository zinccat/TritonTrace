# From: 73_Conv2d_BatchNorm_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_2(
    input_ptr_mean, input_ptr_var, input_ptr_gamma, input_ptr_beta, input_ptr_input,
    output_ptr_normalized, output_ptr_mean, output_ptr_var, output_ptr_gamma, output_ptr_beta,
    kernel_size_0, kernel_size_1, num_elements_x, num_elements_r, XBLOCK: tl.constexpr
):
    num_elements_x = 16
    num_elements_r = 15
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < num_elements_x
    r_index = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_index < num_elements_r
    r1 = r_index
    x0 = x_index

    mean_accumulator = tl.load(input_ptr_mean + (x0 + 16 * r1), r_mask & x_mask, other=0.0)
    var_accumulator = tl.load(input_ptr_var + (x0 + 16 * r1), r_mask & x_mask, other=0.0)
    gamma_accumulator = tl.load(input_ptr_gamma + (x0 + 16 * r1), r_mask & x_mask, other=0.0)

    running_mean = tl.load(input_ptr_beta + (x0), x_mask, eviction_policy='evict_last')
    running_var = tl.load(input_ptr_input + (x0), x_mask, eviction_policy='evict_last')

    mean_broadcast = tl.broadcast_to(mean_accumulator, [XBLOCK, RBLOCK])
    var_broadcast = tl.broadcast_to(var_accumulator, [XBLOCK, RBLOCK])
    gamma_broadcast = tl.broadcast_to(gamma_accumulator, [XBLOCK, RBLOCK])

    mean_selected = tl.where(r_mask & x_mask, mean_broadcast, 0)
    var_selected = tl.where(r_mask & x_mask, var_broadcast, 0)
    gamma_selected = tl.where(r_mask & x_mask, gamma_broadcast, 0)

    mean_moments, var_moments, count = triton_helpers.welford(mean_selected, var_selected, gamma_selected, 1)

    mean_moments_broadcast = mean_moments[:, None]
    var_moments_broadcast = var_moments[:, None]

    normalization_factor = 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + ((-4) * kernel_size_0 * kernel_size_1)
    normalization_factor_float = normalization_factor.to(tl.float32)

    var_normalized = var_moments_broadcast / normalization_factor_float
    epsilon = 1e-05
    var_normalized_eps = var_normalized + epsilon

    inv_sqrt_var = tl.extra.cuda.libdevice.rsqrt(var_normalized_eps)

    scale_factor = (((64 * kernel_size_0 + ((-64) * kernel_size_0 * kernel_size_1) + 16 * kernel_size_0 * kernel_size_1 * kernel_size_1) / 16) / 
                    ((tl.full([], -1.00000000000000, tl.float64)) + 
                     ((64 * kernel_size_0 + ((-64) * kernel_size_0 * kernel_size_1) + 16 * kernel_size_0 * kernel_size_1 * kernel_size_1) / 16)))
    scale_factor_float = scale_factor.to(tl.float32)

    var_scaled = var_normalized * scale_factor_float
    momentum = 0.1
    var_momentum = var_scaled * momentum

    running_mean_momentum = 0.9
    updated_running_mean = running_mean * running_mean_momentum + var_momentum * momentum

    running_var_momentum = 0.9
    updated_running_var = running_var * running_var_momentum + var_momentum * running_mean_momentum

    tl.store(output_ptr_normalized + (x0), inv_sqrt_var, x_mask)
    tl.store(output_ptr_mean + (x0), updated_running_mean, x_mask)
    tl.store(output_ptr_var + (x0), updated_running_var, x_mask)
    tl.store(output_ptr_gamma + (x0), mean_moments_broadcast, x_mask)
    tl.store(output_ptr_beta + (x0), var_moments_broadcast, x_mask)