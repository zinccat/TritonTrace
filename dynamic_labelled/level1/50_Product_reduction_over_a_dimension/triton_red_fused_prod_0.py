# From: 50_Product_reduction_over_a_dimension

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_prod_0red_fused_prod_0(in_ptr0, out_ptr0, kernel_size0, kernel_size1, x_num_elements, r_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x1 = ((x_index // kernel_size0) % 2)
    x0 = (x_index % kernel_size0)
    x2 = x_index // kernel_size1
    _product_accumulator = tl.full([XBLOCK, RBLOCK], 1, tl.float32)
    x4 = x_index
    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r3 = r_index
        tmp0 = r3 + x1 * ((1 + kernel_size0) // 2)
        tmp1 = kernel_size0
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + kernel_size0 * r3 + x2 * kernel_size0 * kernel_size0 + kernel_size0 * x1 * ((1 + kernel_size0) // 2)), rmask & tmp2 & x_mask, eviction_policy='evict_last', other=1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _product_accumulator * tmp4
        _product_accumulator = tl.where(rmask & x_mask, tmp6, _product_accumulator)
    tmp5 = triton_helpers.prod(_product_accumulator, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp5, x_mask)