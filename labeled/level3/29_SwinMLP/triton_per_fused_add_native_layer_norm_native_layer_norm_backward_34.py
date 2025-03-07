# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_34(
    input_grad_ptr, mean_ptr, variance_ptr, input_ptr, grad_output_ptr, output_grad_ptr, output_mean_ptr, output_variance_ptr, xnumel, rnumel
):
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[:]
    r_mask = r_index < rnumel
    r2 = r_index
    x3 = x_index
    x0 = (x_index % 196)
    x1 = x_index // 196

    tmp_grad_input = tl.load(input_grad_ptr + (r2 + 384 * x3), r_mask, other=0.0)
    tmp_mean = tl.load(mean_ptr + (32 * (((4 + ((x0 % 14))) % 7)) + 224 * (((4 + (x0 // 14)) % 7)) + 1568 * (r2 // 32) + 18816 * (((4 + ((x0 % 14))) // 7)) + 56448 * (triton_helpers.div_floor_integer(4 + (x0 // 14), 7)) + 169344 * x1 + ((r2 % 32))), r_mask, other=0.0)
    tmp_variance = tl.load(variance_ptr + (7 * (((4 + (x0 // 14)) % 7)) + 49 * (r2 // 32) + (((4 + ((x0 % 14))) % 7))), r_mask, eviction_policy='evict_last', other=0.0)
    tmp_input = tl.load(input_ptr + (r2), r_mask, eviction_policy='evict_last', other=0.0)
    tmp_grad_output = tl.load(grad_output_ptr + (r2), r_mask, eviction_policy='evict_last', other=0.0)

    tmp_sum_mean_variance = tmp_mean + tmp_variance
    tmp_sum_grad_input = tmp_grad_input + tmp_sum_mean_variance
    tmp_broadcast_sum = tl.broadcast_to(tmp_sum_grad_input, [RBLOCK])
    tl.where(r_mask, tmp_broadcast_sum, 0)
    tmp_broadcast_sum_again = tl.broadcast_to(tmp_sum_grad_input, [RBLOCK])
    tmp_masked_broadcast = tl.where(r_mask, tmp_broadcast_sum_again, 0)
    tmp_sum_masked = triton_helpers.promote_to_tensor(tl.sum(tmp_masked_broadcast, 0))
    tmp_rnumel = tl.full([1], 384, tl.int32)
    tmp_rnumel_float = tmp_rnumel.to(tl.float32)
    tmp_mean_adjusted = tmp_sum_masked / tmp_rnumel_float
    tmp_adjusted_grad_input = tmp_sum_grad_input - tmp_mean_adjusted
    tmp_squared_adjusted = tmp_adjusted_grad_input * tmp_adjusted_grad_input
    tmp_broadcast_squared = tl.broadcast_to(tmp_squared_adjusted, [RBLOCK])
    tmp_masked_squared = tl.where(r_mask, tmp_broadcast_squared, 0)
    tmp_sum_squared = triton_helpers.promote_to_tensor(tl.sum(tmp_masked_squared, 0))
    tmp_variance_adjusted = tmp_sum_grad_input - tmp_mean_adjusted
    tmp_rnumel_float = 384.0
    tmp_variance_mean = tmp_sum_squared / tmp_rnumel_float
    tmp_epsilon = 1e-05
    tmp_variance_epsilon = tmp_variance_mean + tmp_epsilon
    tmp_reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(tmp_variance_epsilon)
    tmp_normalized_grad_input = tmp_variance_adjusted * tmp_reciprocal_sqrt
    tmp_scaled_grad_input = tmp_normalized_grad_input * tmp_input
    tmp_final_grad_output = tmp_scaled_grad_input + tmp_grad_output
    tmp_scale_factor = 0.0026041666666666665
    tmp_scaled_reciprocal_sqrt = tmp_reciprocal_sqrt * tmp_scale_factor

    tl.store(output_grad_ptr + (r2 + 384 * x3), tmp_normalized_grad_input, r_mask)
    tl.store(output_mean_ptr + (r2 + 384 * x3), tmp_final_grad_output, r_mask)
    tl.store(output_variance_ptr + (x3), tmp_scaled_reciprocal_sqrt, None)