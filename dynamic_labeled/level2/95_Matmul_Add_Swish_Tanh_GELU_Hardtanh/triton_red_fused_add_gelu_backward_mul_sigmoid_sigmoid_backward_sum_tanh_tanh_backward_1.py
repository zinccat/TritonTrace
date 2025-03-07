# From: 95_Matmul_Add_Swish_Tanh_GELU_Hardtanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_gelu_backward_mul_sigmoid_sigmoid_backward_sum_tanh_tanh_backward_1(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 512
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = x_index
    input_data2 = tl.load(input_ptr2 + (x0), x_mask, eviction_policy='evict_last')
    accumulated_result = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r1 = r_index
        input_data0 = tl.load(input_ptr0 + (x0 + 512 * r1), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        input_data1 = tl.load(input_ptr1 + (x0 + 512 * r1), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        
        sum_input1_data2 = input_data1 + input_data2
        sigmoid_result = tl.sigmoid(sum_input1_data2)
        product_sigmoid_sum = sigmoid_result * sum_input1_data2
        tanh_result = tl.extra.cuda.libdevice.tanh(product_sigmoid_sum)
        squared_tanh = tanh_result * tanh_result
        one_minus_squared_tanh = 1.0 - squared_tanh
        product_input0_one_minus_squared_tanh = input_data0 * one_minus_squared_tanh
        product_result1 = product_input0_one_minus_squared_tanh * sigmoid_result
        product_result2 = product_input0_one_minus_squared_tanh * sum_input1_data2
        one_minus_sigmoid = 1.0 - sigmoid_result
        product_sigmoid_one_minus_sigmoid = sigmoid_result * one_minus_sigmoid
        product_result3 = product_result2 * product_sigmoid_one_minus_sigmoid
        final_result = product_result1 + product_result3
        
        broadcasted_result = tl.broadcast_to(final_result, [XBLOCK, RBLOCK])
        accumulated_result = accumulated_result + broadcasted_result
        accumulated_result = tl.where(r_mask & x_mask, accumulated_result, accumulated_result)
    
    summed_result = tl.sum(accumulated_result, 1)[:, None]
    tl.store(output_ptr0 + (x0), summed_result, x_mask)