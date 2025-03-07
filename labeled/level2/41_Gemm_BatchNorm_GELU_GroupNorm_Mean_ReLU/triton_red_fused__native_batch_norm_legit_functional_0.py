# From: 41_Gemm_BatchNorm_GELU_GroupNorm_Mean_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_0(
    input_ptr_mean, input_ptr_var, input_ptr_bias, 
    output_ptr_normalized, output_ptr_scale, output_ptr_bias, 
    num_elements, num_reduction_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements = 1024
    num_reduction_elements = 128
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    variance_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)

    for r_offset in range(0, num_reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < num_reduction_elements
        r_indices_flat = r_indices
        input_data = tl.load(
            input_ptr_mean + (x_indices_flat + (1024 * r_indices_flat)), 
            r_mask & x_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        broadcast_input = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        mean_next, variance_next, weight_next = triton_helpers.welford_reduce(
            broadcast_input, mean_accumulator, variance_accumulator, weight_accumulator, r_offset == 0
        )
        mean_accumulator = tl.where(r_mask & x_mask, mean_next, mean_accumulator)
        variance_accumulator = tl.where(r_mask & x_mask, variance_next, variance_accumulator)
        weight_accumulator = tl.where(r_mask & x_mask, weight_next, weight_accumulator)

    mean_final, variance_final, weight_final = triton_helpers.welford(
        mean_accumulator, variance_accumulator, weight_accumulator, 1
    )
    mean_final_expanded = mean_final[:, None]
    variance_final_expanded = variance_final[:, None]
    weight_final_expanded = weight_final[:, None]

    tl.store(output_ptr_normalized + (x_indices_flat), mean_final_expanded, x_mask)
    input_scale = tl.load(input_ptr_var + (x_indices_flat), x_mask, eviction_policy='evict_last')
    input_bias = tl.load(input_ptr_bias + (x_indices_flat), x_mask, eviction_policy='evict_last')

    epsilon = 128.0
    variance_adjusted = variance_final_expanded / epsilon
    epsilon_stable = 1e-05
    variance_adjusted_stable = variance_adjusted + epsilon_stable
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_adjusted_stable)

    scale_factor = 1.0078740157480315
    variance_scaled = variance_adjusted * scale_factor
    momentum = 0.1
    variance_scaled_momentum = variance_scaled * momentum

    scale_momentum = 0.9
    input_scale_momentum = input_scale * scale_momentum
    scale_update = variance_scaled_momentum + input_scale_momentum

    bias_momentum = 0.9
    input_bias_momentum = input_bias * bias_momentum
    bias_update = weight_final_expanded * momentum + input_bias_momentum

    tl.store(output_ptr_scale + (x_indices_flat), inv_sqrt_variance, x_mask)
    tl.store(output_ptr_bias + (x_indices_flat), scale_update, x_mask)
    tl.store(output_ptr_normalized + (x_indices_flat), bias_update, x_mask)