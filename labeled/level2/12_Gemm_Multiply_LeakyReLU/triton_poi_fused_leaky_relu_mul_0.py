# From: 12_Gemm_Multiply_LeakyReLU

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_leaky_relu_mul_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    # Load input data
    input_index = xindex
    input_index_mod = xindex % 512
    input_data0 = tl.load(in_ptr0 + (input_index), None)
    input_data1 = tl.load(in_ptr1 + (input_index_mod), None, eviction_policy='evict_last')
    
    # Perform element-wise addition
    element_sum = input_data0 + input_data1
    
    # Scale the sum
    scale_factor = 2.0
    scaled_sum = element_sum * scale_factor
    
    # Leaky ReLU threshold and negative slope
    threshold = 0.0
    negative_slope = 0.1
    
    # Apply Leaky ReLU
    is_positive = scaled_sum > threshold
    negative_part = scaled_sum * negative_slope
    leaky_relu_output = tl.where(is_positive, scaled_sum, negative_part)
    
    # Store results
    tl.store(out_ptr0 + (input_index), is_positive, None)
    tl.store(out_ptr1 + (input_index), leaky_relu_output, None)