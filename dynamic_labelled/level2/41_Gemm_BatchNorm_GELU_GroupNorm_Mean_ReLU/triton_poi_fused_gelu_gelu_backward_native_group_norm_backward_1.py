# From: 41_Gemm_BatchNorm_GELU_GroupNorm_Mean_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_gelu_gelu_backward_native_group_norm_backward_1(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    xdiv_1024 = xindex // 1024
    xdiv_128 = xindex // 128
    xmod_1024 = xindex % 1024
    xoriginal_index = xindex

    mask0 = tl.load(in_ptr0 + (xdiv_1024), xmask, eviction_policy='evict_last').to(tl.int1)
    mask1 = tl.load(in_ptr1 + (xdiv_1024), xmask, eviction_policy='evict_last')
    group_norm_mean = tl.load(in_ptr2 + (xdiv_128), xmask, eviction_policy='evict_last')
    group_norm_var = tl.load(in_ptr3 + (xmod_1024), xmask, eviction_policy='evict_last')
    input_data = tl.load(in_ptr4 + (xoriginal_index), xmask)
    group_norm_mean_full = tl.load(in_ptr5 + (xdiv_128), xmask, eviction_policy='evict_last')
    group_norm_var_full = tl.load(in_ptr6 + (xdiv_128), xmask, eviction_policy='evict_last')
    group_norm_count = tl.load(in_ptr7 + (xdiv_128), xmask, eviction_policy='evict_last')
    group_norm_mean_full_index = tl.load(in_ptr5 + (xoriginal_index // 128), xmask, eviction_policy='evict_last')
    group_norm_var_full_index = tl.load(in_ptr6 + (xoriginal_index // 128), xmask, eviction_policy='evict_last')
    group_norm_count_index = tl.load(in_ptr7 + (xoriginal_index // 128), xmask, eviction_policy='evict_last')
    group_norm_mean_index = tl.load(in_ptr2 + (xoriginal_index // 128), xmask, eviction_policy='evict_last')

    zero = 0.0
    masked_input = tl.where(mask0, zero, mask1)
    scale_factor = 0.0009765625
    scaled_input = masked_input * scale_factor
    group_norm_std = group_norm_mean * group_norm_var
    scaled_group_norm = scaled_input * group_norm_std
    half = 0.5
    input_half = input_data * half
    sqrt_2 = 0.7071067811865476
    input_scaled = input_data * sqrt_2
    erf_result = tl.extra.cuda.libdevice.erf(input_scaled)
    one = 1.0
    erf_plus_one = erf_result + one
    input_erf_scaled = input_half * erf_plus_one
    group_norm_var_diff = group_norm_mean_full * group_norm_var_full - group_norm_count
    group_norm_var_scaled = group_norm_var_diff * group_norm_mean
    group_norm_var_cubed = group_norm_var_scaled * group_norm_var_scaled * group_norm_var_scaled
    variance_scale = 0.0078125
    variance_scaled = group_norm_var_cubed * variance_scale
    input_erf_scaled_var = input_erf_scaled * variance_scaled
    combined_result = scaled_group_norm + input_erf_scaled_var
    group_norm_var_diff_index = group_norm_mean_full_index * group_norm_var_full_index - group_norm_count_index
    group_norm_var_scaled_index = group_norm_var_diff_index * group_norm_mean_index
    group_norm_var_cubed_index = group_norm_var_scaled_index * group_norm_var_scaled_index * group_norm_var_scaled_index
    variance_scaled_index = group_norm_var_cubed_index * variance_scale
    variance_scaled_neg = -variance_scaled_index
    variance_scaled_neg_var = variance_scaled_neg * group_norm_var_full_index
    group_norm_mean_scaled = group_norm_mean_full_index * variance_scale
    variance_scaled_diff = variance_scaled_neg_var - group_norm_mean_scaled
    final_combined_result = combined_result + variance_scaled_diff
    erf_half = erf_plus_one * half
    input_squared = input_data * input_data
    neg_half = -0.5
    exp_component = tl.math.exp(input_squared * neg_half)
    sqrt_2_pi = 0.3989422804014327
    gaussian_component = exp_component * sqrt_2_pi
    input_gaussian = input_data * gaussian_component
    final_erf_component = erf_half + input_gaussian
    final_result = final_combined_result * final_erf_component
    tl.store(in_out_ptr0 + (xoriginal_index), final_result, xmask)