# From: 89_ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_mul_neg_sigmoid_sigmoid_backward_sub_sum_2red_fused_add_mul_neg_sigmoid_sigmoid_backward_sub_sum_2(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, kernel_size0, kernel_size1, kernel_size2, kernel_size3, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):

    xnumel = 336
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x1 = x_index // 16
    x0 = (x_index % 16)
    temp_result = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = x_index

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r2 = r_index
        temp_index = r2 + x1 * ((20 + kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2) // 21)
        kernel_product = kernel_size0 * kernel_size1 * kernel_size2 * kernel_size2
        index_mask = temp_index < kernel_product

        load0 = tl.load(
            input_ptr0 + (kernel_size1 * x0 * kernel_size2 * kernel_size2 + 
                          16 * kernel_size1 * kernel_size2 * kernel_size2 * 
                          (((temp_index // kernel_size3) % kernel_size0) + 
                           (temp_index % kernel_size3))),
            r_mask & index_mask & x_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        load1 = tl.load(
            input_ptr1 + (kernel_size1 * x0 * kernel_size2 * kernel_size2 + 
                          16 * kernel_size1 * kernel_size2 * kernel_size2 * 
                          (((temp_index // kernel_size3) % kernel_size0) + 
                           (temp_index % kernel_size3))),
            r_mask & index_mask & x_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        load2 = tl.load(
            input_ptr2 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])),
            r_mask & index_mask & x_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        diff = load1 - load2
        sigmoid_val = tl.sigmoid(diff)
        product1 = load0 * sigmoid_val
        product2 = load0 * diff
        one_val = 1.0
        one_minus_sigmoid = one_val - sigmoid_val
        product3 = product2 * one_minus_sigmoid
        sum_products = product1 + product3
        neg_sum = -sum_products

        temp_neg = tl.full(neg_sum.shape, 0, neg_sum.dtype)
        neg_broadcast = tl.where(index_mask, neg_sum, temp_neg)
        neg_broadcast_expanded = tl.broadcast_to(neg_broadcast, [XBLOCK, RBLOCK])
        temp_result = temp_result + neg_broadcast_expanded

        temp_result = tl.where(r_mask & x_mask, temp_result, temp_result)

    sum_result = tl.sum(temp_result, 1)[:, None]
    tl.store(output_ptr0 + (x3), sum_result, x_mask)