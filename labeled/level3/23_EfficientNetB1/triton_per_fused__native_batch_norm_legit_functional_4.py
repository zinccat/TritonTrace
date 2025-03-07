# From: 23_EfficientNetB1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_4(
    input_ptr_mean, input_ptr_var, input_ptr_count, input_ptr_running_mean, input_ptr_running_var,
    output_ptr_normalized, output_ptr_running_mean, output_ptr_running_var, output_ptr_mean, output_ptr_var, 
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 32
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_index = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_index < rnumel
    r1 = r_index
    x0 = x_index

    mean = tl.load(input_ptr_mean + (x0 + 32 * r1), r_mask & x_mask, other=0.0)
    variance = tl.load(input_ptr_var + (x0 + 32 * r1), r_mask & x_mask, other=0.0)
    count = tl.load(input_ptr_count + (x0 + 32 * r1), r_mask & x_mask, other=0.0)
    running_mean = tl.load(input_ptr_running_mean + (x0), x_mask, eviction_policy='evict_last')
    running_var = tl.load(input_ptr_running_var + (x0), x_mask, eviction_policy='evict_last')

    mean_broadcast = tl.broadcast_to(mean, [XBLOCK, RBLOCK])
    variance_broadcast = tl.broadcast_to(variance, [XBLOCK, RBLOCK])
    count_broadcast = tl.broadcast_to(count, [XBLOCK, RBLOCK])

    mean_masked = tl.where(r_mask & x_mask, mean_broadcast, 0)
    variance_masked = tl.where(r_mask & x_mask, variance_broadcast, 0)
    count_masked = tl.where(r_mask & x_mask, count_broadcast, 0)

    mean_accum, variance_accum, count_accum = triton_helpers.welford(mean_masked, variance_masked, count_masked, 1)

    mean_accum_expanded = mean_accum[:, None]
    variance_accum_expanded = variance_accum[:, None]

    variance_scale = 144000.0
    variance_normalized = variance_accum_expanded / variance_scale
    epsilon = 1e-05
    variance_stabilized = variance_normalized + epsilon
    variance_reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(variance_stabilized)

    momentum = 0.1
    mean_momentum = mean_accum_expanded * momentum
    running_mean_momentum = running_mean * 0.9
    updated_running_mean = mean_momentum + running_mean_momentum

    variance_momentum = variance_accum_expanded * 1.00000694449267 * momentum
    updated_running_var = variance_momentum * 0.9 + running_var * 0.9

    tl.store(output_ptr_normalized + (x0), variance_reciprocal_sqrt, x_mask)
    tl.store(output_ptr_running_mean + (x0), updated_running_mean, x_mask)
    tl.store(output_ptr_running_var + (x0), updated_running_var, x_mask)
    tl.store(output_ptr_mean + (x0), mean_accum_expanded, x_mask)
    tl.store(output_ptr_var + (x0), variance_accum_expanded, x_mask)