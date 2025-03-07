# From: 82_Conv2d_Tanh_Scaling_BiasAdd_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_convolution_mul_tanh_0poi_fused_add_convolution_mul_tanh_0(
    in_out_ptr, input_ptr, bias_ptr, output_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    # Calculate the starting index for the current program
    start_index = tl.program_id(0) * XBLOCK
    # Generate a range of indices within the block
    indices = start_index + tl.arange(0, XBLOCK)[:]
    # Create a mask to ensure indices are within bounds
    valid_mask = indices < num_elements

    # Calculate indices for input and output
    output_indices = indices
    input_indices = ((indices // kernel_size) % 16)

    # Load data from input pointers with eviction policy
    output_data = tl.load(in_out_ptr + (output_indices), valid_mask, eviction_policy='evict_last')
    input_data = tl.load(input_ptr + (input_indices), valid_mask, eviction_policy='evict_last')
    bias_data = tl.load(bias_ptr + (input_indices), valid_mask, eviction_policy='evict_last')

    # Perform operations
    added_data = output_data + input_data
    tanh_data = tl.extra.cuda.libdevice.tanh(added_data)
    scaled_data = tanh_data * 2.0
    result_data = scaled_data + bias_data

    # Store results back to memory
    tl.store(in_out_ptr + (output_indices), added_data, valid_mask)
    tl.store(output_ptr + (output_indices), result_data, valid_mask)