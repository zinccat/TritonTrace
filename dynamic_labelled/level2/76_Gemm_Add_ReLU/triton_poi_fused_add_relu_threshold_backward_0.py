# From: 76_Gemm_Add_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_relu_threshold_backward_0poi_fused_add_relu_threshold_backward_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # Calculate the starting index for the current program
    start_index = tl.program_id(0) * XBLOCK
    # Generate a range of indices within the block
    indices = start_index + tl.arange(0, XBLOCK)[:]
    # Create a mask to ensure indices are within bounds
    valid_mask = indices < xnumel
    # Use the same indices for loading and storing
    load_store_indices = indices
    # Calculate the modulo for the second input pointer
    modulo_indices = indices % 512

    # Load data from the input-output pointer with the valid mask
    input_output_data = tl.load(in_out_ptr0 + (load_store_indices), valid_mask)
    # Load data from the input pointer with eviction policy and valid mask
    input_data = tl.load(in_ptr0 + (modulo_indices), valid_mask, eviction_policy='evict_last')
    # Perform element-wise addition
    added_data = input_output_data + input_data
    # Create a tensor filled with zeros for comparison
    zero_tensor = tl.full([1], 0, tl.int32)
    # Apply ReLU operation (maximum of zero and added data)
    relu_output = triton_helpers.maximum(zero_tensor, added_data)
    # Define a threshold value
    threshold_value = 0.0
    # Create a mask for values less than or equal to the threshold
    threshold_mask = relu_output <= threshold_value

    # Store the ReLU output back to the input-output pointer
    tl.store(in_out_ptr0 + (load_store_indices), relu_output, valid_mask)
    # Store the threshold mask to the output pointer
    tl.store(out_ptr0 + (load_store_indices), threshold_mask, valid_mask)