# From: 36_ConvTranspose2d_Min_Sum_GELU_Add

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_convolution_min_0per_fused_convolution_min_0(
    input_ptr0, input_ptr1, output_ptr0, output_ptr1, kernel_size0, kernel_size1, 
    input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < input_num_elements
    r_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_index
    x0 = (x_index % kernel_size0)
    x1 = x_index // kernel_size0
    x3 = x_index
    tmp0 = tl.load(input_ptr0 + (x0 + 4 * r2 * kernel_size1 * kernel_size1 + 64 * x1 * kernel_size1 * kernel_size1), x_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(input_ptr1 + (r2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(x_mask, tmp3, float("inf"))
    tmp6 = triton_helpers.min2(tmp5, 1)[:, None]
    tmp8 = tl.broadcast_to(r_index, tmp5.shape)
    tmp7_val, tmp7_idx = triton_helpers.min_with_index(tmp5, tmp8, 1)
    tmp7 = tmp7_idx[:, None]
    tl.store(output_ptr0 + (x3), tmp6, x_mask)
    tl.store(output_ptr1 + (x3), tmp7, x_mask)