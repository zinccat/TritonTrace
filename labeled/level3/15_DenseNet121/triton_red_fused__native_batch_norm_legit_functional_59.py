# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_59(
    input_ptr_mean, input_ptr_var, input_ptr_gamma, 
    output_ptr_mean, output_ptr_var, output_ptr_gamma, 
    output_ptr_beta, output_ptr_output, 
    total_elements, reduction_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    total_elements = 416
    reduction_elements = 7840
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices

    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)

    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_elements
        r_channel = (r_indices % 784)
        r_sample = r_indices // 784

        input_values = tl.load(
            input_ptr_mean + (r_channel + 784 * x_indices_flat + 326144 * r_sample), 
            r_mask & x_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        broadcasted_values = tl.broadcast_to(input_values, [XBLOCK, RBLOCK])

        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_reduce(
            broadcasted_values, running_mean, running_m2, running_weight, r_offset == 0
        )

        running_mean = tl.where(r_mask & x_mask, running_mean_next, running_mean)
        running_m2 = tl.where(r_mask & x_mask, running_m2_next, running_m2)
        running_weight = tl.where(r_mask & x_mask, running_weight_next, running_weight)

    final_mean, final_var, _ = triton_helpers.welford(
        running_mean, running_m2, running_weight, 1
    )

    final_mean_expanded = final_mean[:, None]
    final_var_expanded = final_var[:, None]

    tl.store(output_ptr_mean + (x_indices_flat), final_mean_expanded, x_mask)
    tl.store(output_ptr_var + (x_indices_flat), final_var_expanded, x_mask)

    input_gamma = tl.load(input_ptr_gamma + (x_indices_flat), x_mask, eviction_policy='evict_last')
    input_beta = tl.load(input_ptr_var + (x_indices_flat), x_mask, eviction_policy='evict_last')

    num_reduction_elements = 7840.0
    epsilon = 1e-05
    variance_adjusted = final_var_expanded / num_reduction_elements + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_adjusted)

    momentum = 0.1
    running_mean_scaled = final_mean_expanded * momentum
    gamma_scaled = input_gamma * (1 - momentum)

    adjusted_mean = running_mean_scaled + gamma_scaled

    beta_momentum = 1.0001275672917465
    beta_scaled = final_var_expanded * beta_momentum * momentum
    beta_adjusted = beta_scaled + input_beta * (1 - momentum)

    tl.store(output_ptr_gamma + (x_indices_flat), inv_stddev, x_mask)
    tl.store(output_ptr_beta + (x_indices_flat), adjusted_mean, x_mask)
    tl.store(output_ptr_output + (x_indices_flat), beta_adjusted, x_mask)