# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_layer_norm_28per_fused_native_layer_norm_28(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[:]
    r_mask = r_index < rnumel
    r1 = r_index
    x0 = x_index
    input_values = tl.load(in_ptr0 + (r1 + 384 * x0), r_mask, other=0.0)
    broadcasted_input = tl.broadcast_to(input_values, [RBLOCK])
    tl.where(r_mask, broadcasted_input, 0)
    broadcasted_input_2 = tl.broadcast_to(broadcasted_input, [RBLOCK])
    masked_broadcasted_input = tl.where(r_mask, broadcasted_input_2, 0)
    sum_of_squares = triton_helpers.promote_to_tensor(tl.sum(masked_broadcasted_input, 0))
    num_elements = tl.full([1], 384, tl.int32)
    num_elements_float = num_elements.to(tl.float32)
    mean = sum_of_squares / num_elements_float
    centered_values = broadcasted_input - mean
    squared_centered_values = centered_values * centered_values
    broadcasted_squared = tl.broadcast_to(squared_centered_values, [RBLOCK])
    masked_squared = tl.where(r_mask, broadcasted_squared, 0)
    sum_of_squared = triton_helpers.promote_to_tensor(tl.sum(masked_squared, 0))
    variance = sum_of_squared / 384.0
    epsilon = 1e-05
    variance_with_epsilon = variance + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), inv_sqrt_variance, None)
    tl.store(out_ptr0 + (x0), mean, None)