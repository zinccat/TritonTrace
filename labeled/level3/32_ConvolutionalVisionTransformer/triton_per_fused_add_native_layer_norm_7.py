# From: 32_ConvolutionalVisionTransformer

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_native_layer_norm_7per_fused_add_native_layer_norm_7(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr
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
    tmp_input0 = tl.load(in_ptr0 + (r1 + 128 * x0), x_mask, other=0.0)
    tmp_input1 = tl.load(in_ptr1 + (r1 + 128 * x0), x_mask, other=0.0)
    tmp_gamma = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp_beta = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp_sum_inputs = tmp_input0 + tmp_input1
    tmp_broadcast_sum = tl.broadcast_to(tmp_sum_inputs, [XBLOCK, RBLOCK])
    tl.where(x_mask, tmp_broadcast_sum, 0)
    tmp_broadcast_sum2 = tl.broadcast_to(tmp_broadcast_sum, [XBLOCK, RBLOCK])
    tmp_masked_broadcast = tl.where(x_mask, tmp_broadcast_sum2, 0)
    tmp_sum_masked = tl.sum(tmp_masked_broadcast, 1)[:, None]
    tmp_rblock_size = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp_rblock_size_float = tmp_rblock_size.to(tl.float32)
    tmp_mean = tmp_sum_masked / tmp_rblock_size_float
    tmp_centered = tmp_broadcast_sum - tmp_mean
    tmp_squared = tmp_centered * tmp_centered
    tmp_broadcast_squared = tl.broadcast_to(tmp_squared, [XBLOCK, RBLOCK])
    tmp_masked_squared = tl.where(x_mask, tmp_broadcast_squared, 0)
    tmp_sum_squared = tl.sum(tmp_masked_squared, 1)[:, None]
    tmp_variance = tmp_sum_squared / 128.0
    epsilon = 1e-05
    tmp_variance_epsilon = tmp_variance + epsilon
    tmp_reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(tmp_variance_epsilon)
    tmp_normalized = tmp_centered * tmp_reciprocal_sqrt
    tmp_scaled = tmp_normalized * tmp_gamma
    tmp_output = tmp_scaled + tmp_beta
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp_reciprocal_sqrt, x_mask)
    tl.store(out_ptr1 + (r1 + 128 * x0), tmp_output, x_mask)
    tl.store(out_ptr0 + (x0), tmp_mean, x_mask)