# From: 89_ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_max_mul_sigmoid_sub_3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, kernel_size0, kernel_size1, kernel_size2, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 16
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    reduction_index_2 = reduction_index
    kernel_index0 = (input_index % kernel_size0)
    kernel_index1 = input_index // kernel_size0
    linear_index = input_index
    temp0 = tl.load(in_ptr0 + (kernel_index0 + kernel_size1 * reduction_index_2 * kernel_size2 * kernel_size2 + 16 * kernel_size1 * kernel_index1 * kernel_size2 * kernel_size2), input_mask, eviction_policy='evict_last', other=0.0)
    temp1 = tl.load(in_ptr1 + (reduction_index_2), None, eviction_policy='evict_last')
    temp2 = temp0 - temp1
    temp3 = tl.sigmoid(temp2)
    temp4 = temp3 * temp2
    temp5 = tl.broadcast_to(temp4, [XBLOCK, RBLOCK])
    temp7 = tl.where(input_mask, temp5, float("-inf"))
    temp8 = triton_helpers.max2(temp7, 1)[:, None]
    temp10 = tl.broadcast_to(reduction_index, temp7.shape)
    temp9_val, temp9_idx = triton_helpers.max_with_index(temp7, temp10, 1)
    temp9 = temp9_idx[:, None]
    tl.store(out_ptr0 + (linear_index), temp8, input_mask)
    tl.store(out_ptr1 + (linear_index), temp9, input_mask)