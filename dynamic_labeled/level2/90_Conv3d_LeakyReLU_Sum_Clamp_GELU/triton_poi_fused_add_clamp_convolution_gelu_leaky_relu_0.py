# From: 90_Conv3d_LeakyReLU_Sum_Clamp_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_clamp_convolution_gelu_leaky_relu_0(
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
    zero = 0.0
    leaky_relu_slope = 0.2
    is_positive = added_data > zero
    leaky_relu_output = tl.where(is_positive, added_data, added_data * leaky_relu_slope)
    
    # Sum with second input
    sum_output = leaky_relu_output + input_data_1
    
    # Clamp operation
    clamp_min = -1.0
    clamp_max = 1.0
    clamped_output = triton_helpers.maximum(sum_output, clamp_min)
    clamped_output = triton_helpers.minimum(clamped_output, clamp_max)
    
    # GELU approximation
    gelu_coefficient = 0.5
    gelu_sqrt_coefficient = 0.7071067811865476
    gelu_clamped = clamped_output * gelu_coefficient
    gelu_erf_input = clamped_output * gelu_sqrt_coefficient
    erf_result = tl.extra.cuda.libdevice.erf(gelu_erf_input)
    gelu_output = gelu_clamped * (erf_result + clamp_max)
    
    # Store results
    tl.store(in_out_ptr0 + (x3), added_data, x_mask)
    tl.store(out_ptr0 + (x3), gelu_output, x_mask)