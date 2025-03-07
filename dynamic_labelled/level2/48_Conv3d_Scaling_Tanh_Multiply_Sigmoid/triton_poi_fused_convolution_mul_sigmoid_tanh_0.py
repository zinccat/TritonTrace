# From: 48_Conv3d_Scaling_Tanh_Multiply_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_mul_sigmoid_tanh_0poi_fused_convolution_mul_sigmoid_tanh_0(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    # Calculate the starting index for the current program
    start_index = tl.program_id(0) * XBLOCK
    # Generate a range of indices within the block
    indices = start_index + tl.arange(0, XBLOCK)[:]
    # Create a mask to ensure indices are within bounds
    valid_mask = indices < num_elements
    # Use indices for loading and storing
    global_indices = indices
    # Calculate the index for input pointers based on kernel size
    input_indices = ((indices // kernel_size) % 16)
    
    # Load data from input pointers with eviction policy
    output_data = tl.load(in_out_ptr0 + (global_indices), valid_mask, eviction_policy='evict_last')
    input_data_0 = tl.load(in_ptr0 + (input_indices), valid_mask, eviction_policy='evict_last')
    input_data_1 = tl.load(in_ptr1 + (input_indices), valid_mask, eviction_policy='evict_last')
    input_data_2 = tl.load(in_ptr2 + (input_indices), valid_mask, eviction_policy='evict_last')
    
    # Perform computations
    intermediate_sum = output_data + input_data_0
    intermediate_product = intermediate_sum * input_data_1
    tanh_result = tl.extra.cuda.libdevice.tanh(intermediate_product)
    scaled_result = tanh_result * input_data_2
    sigmoid_result = tl.sigmoid(scaled_result)
    
    # Store results back to memory
    tl.store(in_out_ptr0 + (global_indices), intermediate_sum, valid_mask)
    tl.store(out_ptr0 + (global_indices), sigmoid_result, valid_mask)