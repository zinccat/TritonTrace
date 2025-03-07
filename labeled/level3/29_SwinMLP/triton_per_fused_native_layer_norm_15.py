# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_layer_norm_15per_fused_native_layer_norm_15(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 7840
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < rnumel
    r1 = r_indices
    x0 = x_indices
    loaded_values = tl.load(in_ptr0 + (r1 + 192 * x0), r_mask & x_mask, other=0.0)
    broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
    tl.where(r_mask & x_mask, broadcasted_values, 0)
    repeated_broadcasted_values = tl.broadcast_to(broadcasted_values, [XBLOCK, RBLOCK])
    masked_broadcasted_values = tl.where(r_mask & x_mask, repeated_broadcasted_values, 0)
    sum_across_rows = tl.sum(masked_broadcasted_values, 1)[:, None]
    rnumel_tensor = tl.full([XBLOCK, 1], 192, tl.int32)
    rnumel_float_tensor = rnumel_tensor.to(tl.float32)
    mean_values = sum_across_rows / rnumel_float_tensor
    centered_values = broadcasted_values - mean_values
    squared_centered_values = centered_values * centered_values
    broadcasted_squared_values = tl.broadcast_to(squared_centered_values, [XBLOCK, RBLOCK])
    masked_squared_values = tl.where(r_mask & x_mask, broadcasted_squared_values, 0)
    sum_of_squares = tl.sum(masked_squared_values, 1)[:, None]
    variance = sum_of_squares / 192.0
    epsilon = 1e-05
    variance_with_epsilon = variance + epsilon
    reciprocal_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), reciprocal_sqrt_variance, x_mask)
    tl.store(out_ptr0 + (x0), mean_values, x_mask)