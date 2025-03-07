# From: 28_VisionTransformer

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8per_fused_add_native_layer_norm_native_layer_norm_backward_8(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, rnumel):

    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 512

    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    r_index = tl.arange(0, RBLOCK)[:]
    
    input_value = tl.load(in_ptr0 + (r_index + 512 * x_index), None)
    in_out_value = tl.load(in_out_ptr0 + (r_index + 512 * x_index), None)
    mean_value = tl.load(in_ptr1 + (r_index), None, eviction_policy='evict_last')
    variance_value = tl.load(in_ptr2 + (r_index), None, eviction_policy='evict_last')
    gamma_value = tl.load(in_ptr3 + (r_index), None, eviction_policy='evict_last')

    sum_value = in_out_value + mean_value
    adjusted_input = input_value + sum_value
    broadcast_adjusted_input = tl.broadcast_to(adjusted_input, [RBLOCK])
    
    sum_broadcast = triton_helpers.promote_to_tensor(tl.sum(broadcast_adjusted_input, 0))
    block_size = tl.full([1], 512, tl.int32).to(tl.float32)
    mean_adjusted = sum_broadcast / block_size
    
    deviation = broadcast_adjusted_input - mean_adjusted
    squared_deviation = deviation * deviation
    broadcast_squared_deviation = tl.broadcast_to(squared_deviation, [RBLOCK])
    
    sum_squared_deviation = triton_helpers.promote_to_tensor(tl.sum(broadcast_squared_deviation, 0))
    variance = (sum_squared_deviation / 512.0) + 1e-05
    inv_std_dev = tl.extra.cuda.libdevice.rsqrt(variance)
    
    normalized_value = deviation * inv_std_dev
    scaled_value = normalized_value * variance_value
    output_value = scaled_value + gamma_value
    
    epsilon_scaled = inv_std_dev * 0.001953125
    
    tl.store(in_out_ptr0 + (r_index + 512 * x_index), normalized_value, None)
    tl.store(out_ptr2 + (r_index + 512 * x_index), output_value, None)
    tl.store(out_ptr3 + (x_index), epsilon_scaled, None)