# From: 41_Gemm_BatchNorm_GELU_GroupNorm_Mean_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_0(
    input_ptr_mean, input_ptr_var, input_ptr_bias, 
    output_ptr_normalized, output_ptr_running_mean, output_ptr_running_var, 
    kernel_size, total_elements, reduction_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    total_elements = 1024
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    batch_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    batch_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    batch_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)

    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_elements
        r_indices_flat = r_indices
        input_data = tl.load(
            input_ptr_mean + (x_indices_flat + 1024 * r_indices_flat), 
            r_mask & x_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        broadcast_input = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        batch_mean_next, batch_m2_next, batch_weight_next = triton_helpers.welford_reduce(
            broadcast_input, batch_mean, batch_m2, batch_weight, r_offset == 0
        )
        batch_mean = tl.where(r_mask & x_mask, batch_mean_next, batch_mean)
        batch_m2 = tl.where(r_mask & x_mask, batch_m2_next, batch_m2)
        batch_weight = tl.where(r_mask & x_mask, batch_weight_next, batch_weight)

    mean, variance, weight = triton_helpers.welford(batch_mean, batch_m2, batch_weight, 1)
    mean_expanded = mean[:, None]
    variance_expanded = variance[:, None]
    weight_expanded = weight[:, None]

    tl.store(output_ptr_normalized + (x_indices_flat), mean_expanded, x_mask)
    running_mean = tl.load(input_ptr_var + (x_indices_flat), x_mask, eviction_policy='evict_last')
    running_var = tl.load(input_ptr_bias + (x_indices_flat), x_mask, eviction_policy='evict_last')
    kernel_size_float = kernel_size.to(tl.float32)
    normalized_mean = variance_expanded / kernel_size_float
    epsilon = 1e-05
    normalized_variance = normalized_mean + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(normalized_variance)
    scale_factor = (((1024 * kernel_size) / 1024) / ((tl.full([], -1.0, tl.float64)) + ((1024 * kernel_size) / 1024))).to(tl.float32)
    normalized_mean_scaled = normalized_mean * scale_factor
    momentum = 0.1
    adjusted_mean = normalized_mean_scaled * momentum
    momentum_factor = 0.9
    running_mean_scaled = running_mean * momentum_factor
    updated_running_mean = adjusted_mean + running_mean_scaled
    normalized_variance_scaled = variance_expanded * momentum
    running_var_scaled = running_var * momentum_factor
    updated_running_var = normalized_variance_scaled + running_var_scaled

    tl.store(output_ptr_running_mean + (x_indices_flat), inv_std, x_mask)
    tl.store(output_ptr_running_var + (x_indices_flat), updated_running_mean, x_mask)
    tl.store(output_ptr_normalized + (x_indices_flat), updated_running_var, x_mask)