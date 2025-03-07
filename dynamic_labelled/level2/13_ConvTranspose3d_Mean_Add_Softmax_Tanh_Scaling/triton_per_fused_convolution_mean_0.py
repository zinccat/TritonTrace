# From: 13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_convolution_mean_0per_fused_convolution_mean_0(
    input_ptr0, input_ptr1, output_ptr0, kernel_size0, kernel_size1, kernel_size2, 
    input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < input_num_elements
    r_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_index
    x3 = (x_index % kernel_size0)
    x4 = x_index // kernel_size0
    x5 = x_index
    tmp0 = tl.load(
        input_ptr0 + (
            x3 + ((-1) * r2) + ((-16) * x4) + 
            ((-64) * x4 * kernel_size2 * kernel_size2) + 
            ((-4) * r2 * kernel_size2 * kernel_size2) + 
            2 * kernel_size1 * r2 + 
            4 * kernel_size2 * r2 + 
            32 * kernel_size1 * x4 + 
            64 * kernel_size2 * x4 + 
            ((-128) * kernel_size1 * kernel_size2 * x4) + 
            ((-8) * kernel_size1 * kernel_size2 * r2) + 
            8 * kernel_size1 * r2 * kernel_size2 * kernel_size2 + 
            128 * kernel_size1 * x4 * kernel_size2 * kernel_size2
        ), 
        x_mask, 
        eviction_policy='evict_last', 
        other=0.0
    )
    tmp1 = tl.load(input_ptr1 + (r2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(x_mask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tl.store(output_ptr0 + (x5), tmp6, x_mask)