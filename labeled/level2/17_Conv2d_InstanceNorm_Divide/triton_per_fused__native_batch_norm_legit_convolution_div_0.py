# From: 17_Conv2d_InstanceNorm_Divide

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused__native_batch_norm_legit_convolution_div_0(
    in_out_ptr, input_ptr, output_ptr, output_ptr2, output_ptr3, xnumel, rnumel
):
    XBLOCK: tl.constexpr = 1
    rnumel = 900
    RBLOCK: tl.constexpr = 1024
    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[:]
    r_mask = r_index < rnumel
    r2 = r_index
    x3 = x_index
    x0 = x_index % 16
    input_value = tl.load(in_out_ptr + (r2 + (900 * x3)), r_mask, other=0.0)
    weight_value = tl.load(input_ptr + (x0), None, eviction_policy='evict_last')
    sum_value = input_value + weight_value
    broadcast_sum = tl.broadcast_to(sum_value, [RBLOCK])
    tl.where(r_mask, broadcast_sum, 0)
    broadcast_sum2 = tl.broadcast_to(broadcast_sum, [RBLOCK])
    masked_sum = tl.where(r_mask, broadcast_sum2, 0)
    mean_value = triton_helpers.promote_to_tensor(tl.sum(masked_sum, 0))
    num_elements = tl.full([1], 900, tl.int32)
    num_elements_float = num_elements.to(tl.float32)
    mean_value_float = mean_value / num_elements_float
    variance_value = sum_value - mean_value_float
    variance_squared = variance_value * variance_value
    broadcast_variance = tl.broadcast_to(variance_squared, [RBLOCK])
    masked_variance = tl.where(r_mask, broadcast_variance, 0)
    variance_sum = triton_helpers.promote_to_tensor(tl.sum(masked_variance, 0))
    variance_mean = variance_sum / 900.0
    epsilon = 1e-05
    variance_mean_epsilon = variance_mean + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_mean_epsilon)
    normalized_value = variance_value * inv_stddev
    scale_factor = 0.5
    scaled_normalized_value = normalized_value * scale_factor
    tl.store(in_out_ptr + (r2 + (900 * x3)), sum_value, r_mask)
    tl.store(output_ptr2 + (r2 + (900 * x3)), scaled_normalized_value, r_mask)
    tl.store(output_ptr3 + (x3), inv_stddev, None)
    tl.store(output_ptr + (x3), mean_value_float, None)