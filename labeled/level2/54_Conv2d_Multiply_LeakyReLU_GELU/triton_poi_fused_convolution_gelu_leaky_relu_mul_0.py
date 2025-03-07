# From: 54_Conv2d_Multiply_LeakyReLU_GELU

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_convolution_gelu_leaky_relu_mul_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    # Calculate indices
    input_index = xindex
    channel_index = (xindex // 900) % 16
    
    # Load data
    in_out_data = tl.load(in_out_ptr0 + (input_index), None)
    input_data_0 = tl.load(in_ptr0 + (channel_index), None, eviction_policy='evict_last')
    input_data_1 = tl.load(in_ptr1 + (channel_index), None, eviction_policy='evict_last')
    
    # Perform operations
    sum_result = in_out_data + input_data_0
    product_result = sum_result * input_data_1
    
    # Leaky ReLU
    zero = 0.0
    leaky_relu_condition = product_result > zero
    leaky_relu_slope = 0.01
    leaky_relu_output = product_result * leaky_relu_slope
    leaky_relu_result = tl.where(leaky_relu_condition, product_result, leaky_relu_output)
    
    # GELU
    gelu_coefficient_1 = 0.5
    gelu_coefficient_2 = 0.7071067811865476
    gelu_intermediate = leaky_relu_result * gelu_coefficient_2
    erf_result = tl.extra.cuda.libdevice.erf(gelu_intermediate)
    gelu_addition = 1.0
    gelu_final = gelu_coefficient_1 * leaky_relu_result * (erf_result + gelu_addition)
    
    # Store results
    tl.store(in_out_ptr0 + (input_index), sum_result, None)
    tl.store(out_ptr0 + (input_index), gelu_final, None)