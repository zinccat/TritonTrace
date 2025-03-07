# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_30(
    input_grad_ptr, input_ptr, scale_ptr, bias_ptr, output_grad_ptr, 
    output_ptr, mean_var_ptr, xnumel, rnumel):

    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[:]
    r_mask = r_index < rnumel
    r_block = r_index
    x_block = x_index
    x_row = x_index % 196
    x_col = x_index // 196

    grad_input = tl.load(input_grad_ptr + (r_block + 384 * x_block), r_mask, other=0.0)
    scale = tl.load(
        input_ptr + (32 * (((x_row % 14) % 7)) + 224 * (((x_row // 14) % 7)) + 
                     1568 * (r_block // 32) + 18816 * (((x_row % 14)) // 7) + 
                     37632 * (x_row // 98) + 75264 * x_col + ((r_block % 32))), 
        r_mask, other=0.0)
    bias = tl.load(
        scale_ptr + (7 * (((x_row // 14) % 7)) + 49 * (r_block // 32) + 
                     ((((x_row % 14)) % 7))), 
        r_mask, eviction_policy='evict_last', other=0.0)
    mean = tl.load(bias_ptr + (r_block), r_mask, eviction_policy='evict_last', other=0.0)
    variance = tl.load(output_grad_ptr + (r_block), r_mask, eviction_policy='evict_last', other=0.0)

    scaled_bias = scale + bias
    grad_input_scaled = grad_input + scaled_bias
    broadcast_grad_input = tl.broadcast_to(grad_input_scaled, [RBLOCK])
    masked_broadcast_grad_input = tl.where(r_mask, broadcast_grad_input, 0)

    sum_grad_input = triton_helpers.promote_to_tensor(tl.sum(masked_broadcast_grad_input, 0))
    rnumel_tensor = tl.full([1], 384, tl.int32)
    rnumel_float = rnumel_tensor.to(tl.float32)
    mean_grad_input = sum_grad_input / rnumel_float

    grad_input_centered = grad_input_scaled - mean_grad_input
    grad_input_centered_squared = grad_input_centered * grad_input_centered
    broadcast_grad_input_centered_squared = tl.broadcast_to(grad_input_centered_squared, [RBLOCK])
    masked_broadcast_grad_input_centered_squared = tl.where(r_mask, broadcast_grad_input_centered_squared, 0)

    sum_grad_input_centered_squared = triton_helpers.promote_to_tensor(tl.sum(masked_broadcast_grad_input_centered_squared, 0))
    variance_centered = grad_input_scaled - mean_grad_input
    rnumel_float_2 = 384.0
    variance = sum_grad_input_centered_squared / rnumel_float_2
    epsilon = 1e-05
    variance_epsilon = variance + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(variance_epsilon)

    grad_input_normalized = variance_centered * inv_std
    grad_scale = grad_input_normalized * mean
    grad_bias = grad_scale + variance

    grad_scale_factor = 0.0026041666666666665
    inv_std_scaled = inv_std * grad_scale_factor

    tl.store(output_ptr + (r_block + 384 * x_block), grad_input_normalized, r_mask)
    tl.store(output_grad_ptr + (r_block + 384 * x_block), grad_bias, r_mask)
    tl.store(mean_var_ptr + (x_block), inv_std_scaled, None)