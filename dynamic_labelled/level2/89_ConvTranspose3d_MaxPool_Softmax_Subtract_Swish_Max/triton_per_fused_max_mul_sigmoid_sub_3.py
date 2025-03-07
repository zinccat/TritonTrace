# From: 89_ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_max_mul_sigmoid_sub_3per_fused_max_mul_sigmoid_sub_3(
    input_ptr0, input_ptr1, output_ptr0, output_ptr1, kernel_size0, kernel_size1, kernel_size2, 
    input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < input_num_elements
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_indices
    x0 = (x_indices % kernel_size0)
    x1 = x_indices // kernel_size0
    x3 = x_indices
    temp0 = tl.load(input_ptr0 + (x0 + kernel_size1 * r2 * kernel_size2 * kernel_size2 + 16 * kernel_size1 * x1 * kernel_size2 * kernel_size2), x_mask, eviction_policy='evict_last', other=0.0)
    temp1 = tl.load(input_ptr1 + (r2), None, eviction_policy='evict_last')
    temp2 = temp0 - temp1
    temp3 = tl.sigmoid(temp2)
    temp4 = temp3 * temp2
    temp5 = tl.broadcast_to(temp4, [XBLOCK, RBLOCK])
    temp7 = tl.where(x_mask, temp5, float("-inf"))
    temp8 = triton_helpers.max2(temp7, 1)[:, None]
    temp10 = tl.broadcast_to(r_indices, temp7.shape)
    temp9_val, temp9_idx = triton_helpers.max_with_index(temp7, temp10, 1)
    temp9 = temp9_idx[:, None]
    tl.store(output_ptr0 + (x3), temp8, x_mask)
    tl.store(output_ptr1 + (x3), temp9, x_mask)