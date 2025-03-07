# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_82(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, xnumel, rnumel
):
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[:]
    r_mask = r_index < rnumel
    r1 = r_index
    x0 = x_index
    input_output = tl.load(in_out_ptr0 + (r1 + 768 * x0), r_mask, other=0.0)
    input0 = tl.load(in_ptr0 + (r1 + 768 * x0), r_mask, other=0.0)
    input1 = tl.load(in_ptr1 + (r1), r_mask, eviction_policy='evict_last', other=0.0)
    input2 = tl.load(in_ptr2 + (r1), r_mask, eviction_policy='evict_last', other=0.0)
    broadcast_input_output = tl.broadcast_to(input_output, [RBLOCK])
    masked_broadcast_input_output = tl.where(r_mask, broadcast_input_output, 0)
    sum_masked_broadcast = triton_helpers.promote_to_tensor(tl.sum(masked_broadcast_input_output, 0))
    num_elements = tl.full([1], 768, tl.int32)
    num_elements_float = num_elements.to(tl.float32)
    mean = sum_masked_broadcast / num_elements_float
    deviation = broadcast_input_output - mean
    squared_deviation = deviation * deviation
    broadcast_squared_deviation = tl.broadcast_to(squared_deviation, [RBLOCK])
    masked_squared_deviation = tl.where(r_mask, broadcast_squared_deviation, 0)
    sum_squared_deviation = triton_helpers.promote_to_tensor(tl.sum(masked_squared_deviation, 0))
    variance = sum_squared_deviation / 768.0
    epsilon = 1e-05
    variance_epsilon = variance + epsilon
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(variance_epsilon)
    normalized_output = (input_output - mean) * reciprocal_sqrt
    scaled_input1 = normalized_output * input1
    layer_norm_output = scaled_input1 + input2
    final_output = input0 + layer_norm_output
    gamma = 0.0013020833333333333
    scaled_reciprocal_sqrt = reciprocal_sqrt * gamma
    tl.store(in_out_ptr0 + (r1 + 768 * x0), normalized_output, r_mask)
    tl.store(out_ptr2 + (r1 + 768 * x0), final_output, r_mask)
    tl.store(out_ptr3 + (x0), scaled_reciprocal_sqrt, None)