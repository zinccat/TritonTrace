# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_46(
    input_grad_ptr, input_ptr, mean_ptr, variance_ptr, gamma_ptr, output_grad_ptr, output_ptr, output_scale_ptr, xnumel, rnumel
):
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[:]
    r_mask = r_index < rnumel
    r2 = r_index
    x3 = x_index
    x0 = (x_index % 49)
    x1 = x_index // 49

    grad_input = tl.load(input_grad_ptr + (r2 + 768 * x3), r_mask, other=0.0)
    input_value = tl.load(input_ptr + (32 * x0 + 1568 * (r2 // 32) + 37632 * x1 + ((r2 % 32))), r_mask, other=0.0)
    mean_value = tl.load(mean_ptr + (x0 + 49 * (r2 // 32)), r_mask, eviction_policy='evict_last', other=0.0)
    variance_value = tl.load(variance_ptr + (r2), r_mask, eviction_policy='evict_last', other=0.0)
    gamma_value = tl.load(gamma_ptr + (r2), r_mask, eviction_policy='evict_last', other=0.0)

    normalized_input = input_value + mean_value
    input_grad_sum = grad_input + normalized_input
    broadcast_grad_sum = tl.broadcast_to(input_grad_sum, [RBLOCK])
    masked_broadcast_grad_sum = tl.where(r_mask, broadcast_grad_sum, 0)

    sum_grad = triton_helpers.promote_to_tensor(tl.sum(masked_broadcast_grad_sum, 0))
    num_elements = tl.full([1], 768, tl.int32)
    num_elements_float = num_elements.to(tl.float32)
    mean_grad = sum_grad / num_elements_float

    grad_centered = input_grad_sum - mean_grad
    grad_centered_squared = grad_centered * grad_centered
    broadcast_grad_centered_squared = tl.broadcast_to(grad_centered_squared, [RBLOCK])
    masked_broadcast_grad_centered_squared = tl.where(r_mask, broadcast_grad_centered_squared, 0)

    sum_grad_centered_squared = triton_helpers.promote_to_tensor(tl.sum(masked_broadcast_grad_centered_squared, 0))
    variance_centered = grad_centered - mean_grad
    variance_sum = sum_grad_centered_squared / 768.0
    epsilon = 1e-05
    variance_normalized = variance_sum + epsilon
    variance_reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(variance_normalized)

    grad_normalized = variance_centered * variance_reciprocal_sqrt
    grad_scaled = grad_normalized * variance_value
    output_grad = grad_scaled + gamma_value

    scale_factor = 0.0013020833333333333
    output_scale = variance_reciprocal_sqrt * scale_factor

    tl.store(output_grad_ptr + (r2 + 768 * x3), grad_normalized, r_mask)
    tl.store(output_ptr + (r2 + 768 * x3), output_grad, r_mask)
    tl.store(output_scale_ptr + (x3), output_scale, None)