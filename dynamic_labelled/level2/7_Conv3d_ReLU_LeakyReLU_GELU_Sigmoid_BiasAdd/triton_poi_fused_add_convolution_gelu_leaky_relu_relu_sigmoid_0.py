# From: 7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_convolution_gelu_leaky_relu_relu_sigmoid_0poi_fused_add_convolution_gelu_leaky_relu_relu_sigmoid_0(
    in_out_ptr, input_ptr1, input_ptr2, output_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_indices < num_elements
    x3 = x_indices
    x1 = ((x_indices // kernel_size) % 16)
    
    tmp_in_out = tl.load(in_out_ptr + (x3), x_mask, eviction_policy='evict_last')
    tmp_input1 = tl.load(input_ptr1 + (x1), x_mask, eviction_policy='evict_last')
    tmp_input2 = tl.load(input_ptr2 + (x1), x_mask, eviction_policy='evict_last')
    
    tmp_sum = tmp_in_out + tmp_input1
    tmp_zero = tl.full([1], 0, tl.int32)
    tmp_max = triton_helpers.maximum(tmp_zero, tmp_sum)
    
    tmp_leaky_relu_threshold = 0.0
    tmp_leaky_relu_condition = tmp_max > tmp_leaky_relu_threshold
    tmp_leaky_relu_slope = 0.01
    tmp_leaky_relu = tmp_max * tmp_leaky_relu_slope
    tmp_gelu_input = tl.where(tmp_leaky_relu_condition, tmp_max, tmp_leaky_relu)
    
    tmp_gelu_half = 0.5
    tmp_gelu_sqrt2_over2 = 0.7071067811865476
    tmp_gelu_erf_input = tmp_gelu_input * tmp_gelu_sqrt2_over2
    tmp_gelu_erf = tl.extra.cuda.libdevice.erf(tmp_gelu_erf_input)
    tmp_gelu_one = 1.0
    tmp_gelu = tmp_gelu_half * (tmp_gelu_erf + tmp_gelu_one)
    
    tmp_sigmoid = tl.sigmoid(tmp_gelu)
    tmp_final_output = tmp_sigmoid + tmp_input2
    
    tl.store(in_out_ptr + (x3), tmp_sum, x_mask)
    tl.store(output_ptr + (x3), tmp_final_output, x_mask)