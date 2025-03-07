# From: 48_Conv3d_Scaling_Tanh_Multiply_Sigmoid

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_convolution_mul_sigmoid_tanh_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = x_index
    x1 = (x_index // 12600) % 16
    
    # Load data from input pointers
    input_out_value = tl.load(in_out_ptr0 + (x3), None)
    input_value_0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    input_value_1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    input_value_2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    
    # Perform computations
    intermediate_sum = input_out_value + input_value_0
    intermediate_product = intermediate_sum * input_value_1
    tanh_result = tl.extra.cuda.libdevice.tanh(intermediate_product)
    sigmoid_input = tanh_result * input_value_2
    sigmoid_result = tl.sigmoid(sigmoid_input)
    
    # Store results back to output pointers
    tl.store(in_out_ptr0 + (x3), intermediate_sum, None)
    tl.store(out_ptr0 + (x3), sigmoid_result, None)