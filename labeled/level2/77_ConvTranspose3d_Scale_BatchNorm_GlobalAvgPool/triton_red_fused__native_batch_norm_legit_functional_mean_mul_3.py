# From: 77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool

import triton
import triton.language as tl


@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_mean_mul_3(
    output_ptr, input_ptr, mean_ptr, variance_ptr, scale_ptr, bias_ptr, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 512
    rnumel = 20808
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x3 = x_index
    x0 = x_index % 32

    mean_value = tl.load(mean_ptr + (x0), x_mask, eviction_policy='evict_last')
    variance_value = tl.load(variance_ptr + (x0), x_mask, eviction_policy='evict_last')
    scale_value = tl.load(scale_ptr + (x0), x_mask, eviction_policy='evict_last')
    bias_value = tl.load(bias_ptr + (x0), x_mask, eviction_policy='evict_last')

    accumulated_result = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r2 = r_index

        input_value = tl.load(input_ptr + (r2 + (20808 * x3)), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = 2.0
        scaled_input = input_value * tmp1
        centered_input = scaled_input - mean_value

        variance_denominator = 332928.0
        normalized_variance = variance_value / variance_denominator
        epsilon = 1e-05
        adjusted_variance = normalized_variance + epsilon
        inv_std_dev = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)

        normalized_input = centered_input * inv_std_dev
        scaled_normalized_input = normalized_input * scale_value
        biased_output = scaled_normalized_input + bias_value

        broadcasted_output = tl.broadcast_to(biased_output, [XBLOCK, RBLOCK])
        accumulated_result = accumulated_result + broadcasted_output

        accumulated_result = tl.where(r_mask & x_mask, accumulated_result, accumulated_result)

    summed_result = tl.sum(accumulated_result, 1)[:, None]
    normalization_factor = 20808.0
    final_result = summed_result / normalization_factor

    tl.debug_barrier()
    tl.store(output_ptr + (x3), final_result, x_mask)