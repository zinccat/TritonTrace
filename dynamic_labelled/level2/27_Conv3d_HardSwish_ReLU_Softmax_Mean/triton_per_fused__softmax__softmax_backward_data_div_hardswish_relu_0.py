# From: 27_Conv3d_HardSwish_ReLU_Softmax_Mean

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax__softmax_backward_data_div_hardswish_relu_0(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, output_ptr0, kernel_size0, kernel_size1, kernel_size2, 
    x_num_elements, r_num_elements, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    r_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_block_index = r_index
    x_block_index = x_index // kernel_size0
    x_kernel_index = (x_index % kernel_size0)
    x_full_index = x_index

    temp0 = tl.load(input_ptr0 + (r_block_index + 16 * x_block_index), x_mask, eviction_policy='evict_last', other=0.0)
    temp4 = tl.load(
        input_ptr1 + (
            x_kernel_index + 
            ((-128) * x_block_index) + 
            ((-8) * r_block_index) + 
            ((-32) * x_block_index * kernel_size2 * kernel_size2) + 
            ((-2) * r_block_index * kernel_size2 * kernel_size2) + 
            4 * kernel_size1 * r_block_index + 
            8 * kernel_size2 * r_block_index + 
            64 * kernel_size1 * x_block_index + 
            128 * kernel_size2 * x_block_index + 
            kernel_size1 * r_block_index * kernel_size2 * kernel_size2 + 
            ((-64) * kernel_size1 * kernel_size2 * x_block_index) + 
            ((-4) * kernel_size1 * kernel_size2 * r_block_index) + 
            16 * kernel_size1 * x_block_index * kernel_size2 * kernel_size2
        ), 
        x_mask, 
        eviction_policy='evict_last', 
        other=0.0
    )
    temp16 = tl.load(input_ptr2 + (x_full_index), x_mask, eviction_policy='evict_last')
    temp19 = tl.load(input_ptr3 + (x_full_index), x_mask, eviction_policy='evict_last')

    temp1 = kernel_size0
    temp2 = temp1.to(tl.float32)
    temp3 = temp0 / temp2

    temp5 = 3.0
    temp6 = temp4 + temp5
    temp7 = 0.0
    temp8 = triton_helpers.maximum(temp6, temp7)
    temp9 = 6.0
    temp10 = triton_helpers.minimum(temp8, temp9)
    temp11 = temp4 * temp10
    temp12 = 0.16666666666666666
    temp13 = temp11 * temp12
    temp14 = tl.full([1, 1], 0, tl.int32)
    temp15 = triton_helpers.maximum(temp14, temp13)

    temp17 = temp15 - temp16
    temp18 = tl.math.exp(temp17)
    temp20 = temp18 / temp19
    temp21 = temp3 * temp20
    temp22 = tl.broadcast_to(temp21, [XBLOCK, RBLOCK])
    temp24 = tl.where(x_mask, temp22, 0)
    temp25 = tl.sum(temp24, 1)[:, None]

    tl.store(output_ptr0 + (x_full_index), temp25, x_mask)