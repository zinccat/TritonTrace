# From: 74_ConvTranspose3d_LeakyReLU_Multiply_LeakyReLU_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_leaky_relu_mul_0poi_fused_convolution_leaky_relu_mul_0(
    in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, kernel_size, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    # Calculate indices
    global_index = xindex
    channel_index = ((xindex // kernel_size) % 32)
    
    # Load data
    in_out_value = tl.load(in_out_ptr0 + (global_index), None, eviction_policy='evict_last')
    input_value0 = tl.load(in_ptr0 + (channel_index), None, eviction_policy='evict_last')
    input_value1 = tl.load(in_ptr1 + (channel_index), None, eviction_policy='evict_last')
    
    # Perform operations
    sum_value = in_out_value + input_value0
    leaky_relu_threshold = 0.0
    is_positive = sum_value > leaky_relu_threshold
    negative_slope = 0.2
    negative_value = sum_value * negative_slope
    leaky_relu_output = tl.where(is_positive, sum_value, negative_value)
    
    multiplied_value = leaky_relu_output * input_value1
    is_positive_multiplied = multiplied_value > leaky_relu_threshold
    negative_value_multiplied = multiplied_value * negative_slope
    final_output = tl.where(is_positive_multiplied, multiplied_value, negative_value_multiplied)
    
    # Store results
    tl.store(in_out_ptr0 + (global_index), sum_value, None)
    tl.store(out_ptr0 + (global_index), final_output, None)