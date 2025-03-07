# From: 32_ConvolutionalVisionTransformer

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 20
    RBLOCK: tl.constexpr = 128
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = r_index
    x0 = x_index
    input_tensor = tl.load(in_ptr0 + (r1 + 128 * x0), x_mask, other=0.0)
    grad_output = tl.load(in_out_ptr0 + (r1 + 128 * x0), x_mask, other=0.0)
    grad_input = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    mean = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    variance = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    
    grad_output_with_input = grad_output + grad_input
    input_with_grad = input_tensor + grad_output_with_input
    broadcast_input_with_grad = tl.broadcast_to(input_with_grad, [XBLOCK, RBLOCK])
    masked_broadcast_input = tl.where(x_mask, broadcast_input_with_grad, 0)
    
    sum_grad_output_with_input = tl.sum(masked_broadcast_input, 1)[:, None]
    block_size = tl.full([XBLOCK, 1], 128, tl.int32).to(tl.float32)
    mean_grad = sum_grad_output_with_input / block_size
    
    normalized_grad = input_with_grad - mean_grad
    squared_grad = normalized_grad * normalized_grad
    broadcast_squared_grad = tl.broadcast_to(squared_grad, [XBLOCK, RBLOCK])
    masked_broadcast_squared_grad = tl.where(x_mask, broadcast_squared_grad, 0)
    
    sum_squared_grad = tl.sum(masked_broadcast_squared_grad, 1)[:, None]
    variance_grad = (input_with_grad - mean_grad) * (1 / tl.sqrt(sum_squared_grad / 128.0 + 1e-05))
    scaled_variance_grad = variance_grad * mean
    
    final_output = scaled_variance_grad + variance
    epsilon = 0.0078125
    inv_std = 1 / tl.sqrt(sum_squared_grad / 128.0 + 1e-05)
    scaled_inv_std = inv_std * epsilon
    
    tl.store(in_out_ptr0 + (r1 + 128 * x0), variance_grad, x_mask)
    tl.store(out_ptr2 + (r1 + 128 * x0), final_output, x_mask)
    tl.store(out_ptr3 + (x0), scaled_inv_std, x_mask)