# From: 7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_convolution_gelu_leaky_relu_relu_sigmoid_0(
    in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < num_elements
    x3 = x_index
    x1 = ((x_index // kernel_size) % 16)
    
    # Load data with eviction policy
    input_output_data = tl.load(in_out_ptr0 + (x3), x_mask, eviction_policy='evict_last')
    input_data_0 = tl.load(in_ptr0 + (x1), x_mask, eviction_policy='evict_last')
    input_data_1 = tl.load(in_ptr1 + (x1), x_mask, eviction_policy='evict_last')
    
    # Perform addition
    added_data = input_output_data + input_data_0
    
    # Leaky ReLU
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_max = tl.maximum(zero_tensor, added_data)
    leaky_relu_threshold = 0.0
    is_greater = relu_max > leaky_relu_threshold
    leaky_relu_slope = 0.01
    leaky_relu_result = tl.where(is_greater, relu_max, relu_max * leaky_relu_slope)
    
    # GELU
    gelu_coefficient = 0.5
    gelu_coefficient_2 = 0.7071067811865476
    gelu_result = gelu_coefficient * (leaky_relu_result * (tl.extra.cuda.libdevice.erf(leaky_relu_result * gelu_coefficient_2) + 1.0))
    sigmoid_result = tl.sigmoid(gelu_result)
    
    # Final addition and store results
    final_result = sigmoid_result + input_data_1
    tl.store(in_out_ptr0 + (x3), added_data, x_mask)
    tl.store(out_ptr0 + (x3), final_result, x_mask)