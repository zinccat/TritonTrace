# From: 35_GroupNorm_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_2poi_fused_native_group_norm_2(
    input_ptr_mean, input_ptr_inv_std, input_ptr_var, input_ptr_gamma, input_ptr_beta,
    output_ptr, kernel_size_h, kernel_size_w, num_elements, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < num_elements
    x_linear_index = x_index
    x_group_index = x_index // kernel_size_h
    x_channel_index = (x_index // kernel_size_h) % 64
    x_within_kernel_index = x_index % kernel_size_w
    x_channel_group_index = x_index // kernel_size_w

    mean = tl.load(input_ptr_mean + (x_linear_index), x_mask, eviction_policy='evict_last')
    inv_std = tl.load(input_ptr_inv_std + (x_group_index // 8), x_mask, eviction_policy='evict_last')
    var = tl.load(input_ptr_var + (x_group_index // 8), x_mask, eviction_policy='evict_last')
    gamma = tl.load(input_ptr_gamma + (x_channel_index), x_mask, eviction_policy='evict_last')
    beta = tl.load(input_ptr_beta + (x_channel_index), x_mask, eviction_policy='evict_last')

    normalized = mean - inv_std
    var_scaled = 8 * kernel_size_h
    var_scaled_float = var_scaled.to(tl.float32)
    normalized_var = var / var_scaled_float
    epsilon = 1e-05
    rsqrt_var = normalized_var + epsilon
    inv_std_dev = tl.extra.cuda.libdevice.rsqrt(rsqrt_var)
    scaled_normalized = normalized * inv_std_dev
    scaled_gamma = scaled_normalized * gamma
    output = scaled_gamma + beta

    tl.store(output_ptr + (x_within_kernel_index + x_channel_group_index * (kernel_size_h // kernel_size_w)), output, x_mask)