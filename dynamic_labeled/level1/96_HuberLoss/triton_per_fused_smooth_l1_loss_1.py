# From: 96_HuberLoss

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_smooth_l1_loss_1(in_out_ptr0, in_ptr0, kernel_size0, kernel_size1, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    mask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_indices = tl.arange(0, RBLOCK)[None, :]
    mask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    row_indices = r_indices
    loaded_values = tl.load(in_ptr0 + (row_indices), None)
    broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
    summed_values = tl.sum(broadcasted_values, 1)[:, None]
    kernel_product = kernel_size0 * kernel_size1
    kernel_product_float = kernel_product.to(tl.float32)
    result = summed_values / kernel_product_float
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), result, None)