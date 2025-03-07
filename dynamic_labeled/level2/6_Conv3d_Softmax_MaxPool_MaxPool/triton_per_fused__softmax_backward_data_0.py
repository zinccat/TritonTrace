# From: 6_Conv3d_Softmax_MaxPool_MaxPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_backward_data_0(
    input_grad_ptr, output_grad_ptr, output_ptr, kernel_size_d, kernel_size_h, kernel_size_w, num_elements_x, num_elements_r, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < num_elements_x
    r_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_index
    x3 = (x_index % kernel_size_d)
    x4 = x_index // kernel_size_d
    x5 = x_index

    input_grad = tl.load(
        input_grad_ptr + (
            x3 + ((-128) * x4) + ((-8) * r2) + ((-32) * x4 * kernel_size_w * kernel_size_w) +
            ((-2) * r2 * kernel_size_w * kernel_size_w) + 4 * kernel_size_h * r2 +
            8 * kernel_size_w * r2 + 64 * kernel_size_h * x4 + 128 * kernel_size_w * x4 +
            kernel_size_h * r2 * kernel_size_w * kernel_size_w + ((-64) * kernel_size_h * kernel_size_w * x4) +
            ((-4) * kernel_size_h * kernel_size_w * r2) + 16 * kernel_size_h * x4 * kernel_size_w * kernel_size_w
        ),
        x_mask,
        eviction_policy='evict_last',
        other=0.0
    )

    output_grad = tl.load(
        output_grad_ptr + (
            x3 + ((-128) * x4) + ((-8) * r2) + ((-32) * x4 * kernel_size_w * kernel_size_w) +
            ((-2) * r2 * kernel_size_w * kernel_size_w) + 4 * kernel_size_h * r2 +
            8 * kernel_size_w * r2 + 64 * kernel_size_h * x4 + 128 * kernel_size_w * x4 +
            kernel_size_h * r2 * kernel_size_w * kernel_size_w + ((-64) * kernel_size_h * kernel_size_w * x4) +
            ((-4) * kernel_size_h * kernel_size_w * r2) + 16 * kernel_size_h * x4 * kernel_size_w * kernel_size_w
        ),
        x_mask,
        eviction_policy='evict_last',
        other=0.0
    )

    elementwise_product = input_grad * output_grad
    broadcasted_product = tl.broadcast_to(elementwise_product, [XBLOCK, RBLOCK])
    masked_product = tl.where(x_mask, broadcasted_product, 0)
    sum_over_r = tl.sum(masked_product, 1)[:, None]

    tl.store(output_ptr + (x5), sum_over_r, x_mask)