# From: 61_ConvTranspose3d_ReLU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_1(input_ptr_mean, input_ptr_var, input_ptr_count, output_ptr_mean, output_ptr_var, output_ptr_count, kernel_size_0, kernel_size_1, num_elements_x, num_elements_r, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 4
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < num_elements_x
    r_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = r_index
    x0 = x_index
    mean0 = tl.load(input_ptr_mean + (r1 + 4*x0), x_mask, other=0.0)
    mean1 = tl.load(input_ptr_var + (r1 + 4*x0), x_mask, other=0.0)
    mean2 = tl.load(input_ptr_count + (r1 + 4*x0), x_mask, other=0.0)
    broadcast_mean0 = tl.broadcast_to(mean0, [XBLOCK, RBLOCK])
    broadcast_mean1 = tl.broadcast_to(mean1, [XBLOCK, RBLOCK])
    broadcast_mean2 = tl.broadcast_to(mean2, [XBLOCK, RBLOCK])
    masked_mean0 = tl.where(x_mask, broadcast_mean0, 0)
    masked_mean1 = tl.where(x_mask, broadcast_mean1, 0)
    masked_mean2 = tl.where(x_mask, broadcast_mean2, 0)
    mean_accum, var_accum, count_accum = triton_helpers.welford(masked_mean0, masked_mean1, masked_mean2, 1)
    mean_accum_expanded = mean_accum[:, None]
    var_accum_expanded = var_accum[:, None]
    count_accum_expanded = count_accum[:, None]
    normalization_factor = 128 + 32*kernel_size_0*kernel_size_0 + 64*kernel_size_1 + 128*kernel_size_0 + 16*kernel_size_1*kernel_size_0*kernel_size_0 + 64*kernel_size_0*kernel_size_1
    normalization_factor_float = normalization_factor.to(tl.float32)
    normalized_variance = var_accum_expanded / normalization_factor_float
    epsilon = 1e-05
    variance_with_epsilon = normalized_variance + epsilon
    reciprocal_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    tl.store(output_ptr_count + (x0), reciprocal_sqrt_variance, x_mask)
    tl.store(output_ptr_mean + (x0), mean_accum_expanded, x_mask)
    tl.store(output_ptr_var + (x0), var_accum_expanded, x_mask)