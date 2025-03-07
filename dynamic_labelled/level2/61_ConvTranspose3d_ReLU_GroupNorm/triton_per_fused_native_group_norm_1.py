# From: 61_ConvTranspose3d_ReLU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_1per_fused_native_group_norm_1(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, output_ptr1, output_ptr2, kernel_size0, kernel_size1, 
    num_elements_x, num_elements_r, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 4
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_group = r_indices
    x_group = x_indices

    input0 = tl.load(input_ptr0 + (r_group + 4 * x_group), x_mask, other=0.0)
    input1 = tl.load(input_ptr1 + (r_group + 4 * x_group), x_mask, other=0.0)
    input2 = tl.load(input_ptr2 + (r_group + 4 * x_group), x_mask, other=0.0)

    broadcast_input0 = tl.broadcast_to(input0, [XBLOCK, RBLOCK])
    broadcast_input1 = tl.broadcast_to(input1, [XBLOCK, RBLOCK])
    broadcast_input2 = tl.broadcast_to(input2, [XBLOCK, RBLOCK])

    masked_input0 = tl.where(x_mask, broadcast_input0, 0)
    masked_input1 = tl.where(x_mask, broadcast_input1, 0)
    masked_input2 = tl.where(x_mask, broadcast_input2, 0)

    mean, variance, _ = triton_helpers.welford(masked_input0, masked_input1, masked_input2, 1)

    mean_expanded = mean[:, None]
    variance_expanded = variance[:, None]

    normalization_factor = 128 + 32 * kernel_size0 * kernel_size0 + 64 * kernel_size1 + 128 * kernel_size0 + 16 * kernel_size1 * kernel_size0 * kernel_size0 + 64 * kernel_size0 * kernel_size1
    normalization_factor_float = normalization_factor.to(tl.float32)

    normalized_variance = variance_expanded / normalization_factor_float
    epsilon = 1e-05
    adjusted_variance = normalized_variance + epsilon

    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)

    tl.store(output_ptr2 + (x_group), reciprocal_sqrt, x_mask)
    tl.store(output_ptr0 + (x_group), mean_expanded, x_mask)
    tl.store(output_ptr1 + (x_group), variance_expanded, x_mask)