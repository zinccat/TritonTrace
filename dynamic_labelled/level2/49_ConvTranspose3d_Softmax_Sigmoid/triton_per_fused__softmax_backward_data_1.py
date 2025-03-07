# From: 49_ConvTranspose3d_Softmax_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_backward_data_1per_fused__softmax_backward_data_1(
    input_ptr, output_ptr, kernel_size_0, kernel_size_1, x_num_elements, r_num_elements, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_indices
    x0 = (x_indices % kernel_size_0)
    x1 = x_indices // kernel_size_0
    x3 = x_indices
    tmp0 = tl.load(input_ptr + (x0 + 8192 * kernel_size_1 * r2 + 524288 * kernel_size_1 * x1), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tl.store(output_ptr + (x3), tmp3, None)