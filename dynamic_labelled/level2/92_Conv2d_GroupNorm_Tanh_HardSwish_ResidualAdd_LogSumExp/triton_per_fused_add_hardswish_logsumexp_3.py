# From: 92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_hardswish_logsumexp_3(
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
    x3 = (x_index % kernel_size0)
    x4 = x_index // kernel_size0
    x5 = x_index

    input_offset = (
        x3 + 4 * r2 + 64 * x4 + r2 * kernel_size1 * kernel_size1 + 
        (-64) * kernel_size1 * x4 + (-4) * kernel_size1 * r2 + 
        16 * x4 * kernel_size1 * kernel_size1
    )
    
    tmp0 = tl.load(input_ptr0 + input_offset, x_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(input_ptr1 + input_offset, x_mask, eviction_policy='evict_last', other=0.0)
    
    tmp2 = 3.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = 6.0
    tmp7 = triton_helpers.minimum(tmp5, tmp6)
    tmp8 = tmp1 * tmp7
    tmp9 = 0.16666666666666666
    tmp10 = tmp8 * tmp9
    tmp11 = tmp0 + tmp10
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(x_mask, tmp12, float("-inf"))
    tmp15 = triton_helpers.max2(tmp14, 1)[:, None]
    tmp16 = tl.math.abs(tmp15)
    tmp17 = float("inf")
    tmp18 = tmp16 == tmp17
    tmp19 = tl.where(tmp18, tmp4, tmp15)
    tmp20 = tmp11 - tmp19
    tmp21 = tl.math.exp(tmp20)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(x_mask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    
    tl.store(output_ptr0 + (x5), tmp15, x_mask)
    tl.store(output_ptr1 + (x5), tmp25, x_mask)