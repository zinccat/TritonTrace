# From: 52_Conv2d_Activation_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_mul_softplus_tanh_1(
    input_ptr, output_mean_ptr, output_variance_ptr, output_count_ptr, 
    num_elements, num_reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements = 240
    num_reduction_elements = 7680
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_channel = x_indices % 16
    x_feature = (x_indices // 16)
    tmp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_linear_index = x_indices

    for r_offset in range(0, num_reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < num_reduction_elements
        r_linear_index = r_indices
        input_index = (
            30 * (((r_linear_index + (7680 * x_feature)) // 30) % 30) 
            + (900 * x_channel) 
            + (14400 * ((r_linear_index + (7680 * x_feature)) // 900)) 
            + (r_linear_index % 30)
        )
        input_data = tl.load(input_ptr + input_index, r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        threshold = 20.0
        is_greater_than_threshold = input_data > threshold
        exp_input = tl.math.exp(input_data)
        log1p_exp_input = tl.extra.cuda.libdevice.log1p(exp_input)
        softplus_input = tl.where(is_greater_than_threshold, input_data, log1p_exp_input)
        tanh_softplus = tl.extra.cuda.libdevice.tanh(softplus_input)
        weighted_input = tanh_softplus * input_data
        broadcasted_weighted_input = tl.broadcast_to(weighted_input, [XBLOCK, RBLOCK])
        
        tmp_mean_next, tmp_m2_next, tmp_weight_next = triton_helpers.welford_reduce(
            broadcasted_weighted_input, tmp_mean, tmp_m2, tmp_weight, r_offset == 0
        )
        tmp_mean = tl.where(r_mask & x_mask, tmp_mean_next, tmp_mean)
        tmp_m2 = tl.where(r_mask & x_mask, tmp_m2_next, tmp_m2)
        tmp_weight = tl.where(r_mask & x_mask, tmp_weight_next, tmp_weight)

    mean, variance, count = triton_helpers.welford(tmp_mean, tmp_m2, tmp_weight, 1)
    mean_broadcasted = mean[:, None]
    variance_broadcasted = variance[:, None]
    count_broadcasted = count[:, None]

    tl.store(output_mean_ptr + (x_linear_index), mean_broadcasted, x_mask)
    tl.store(output_variance_ptr + (x_linear_index), variance_broadcasted, x_mask)
    tl.store(output_count_ptr + (x_linear_index), count_broadcasted, x_mask)