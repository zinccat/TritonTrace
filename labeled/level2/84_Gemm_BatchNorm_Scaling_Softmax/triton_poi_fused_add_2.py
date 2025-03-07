# From: 84_Gemm_BatchNorm_Scaling_Softmax

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_add_2(in_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    # Calculate the starting offset for the current program ID
    start_offset = tl.program_id(0) * XBLOCK
    # Generate a range of indices from 0 to XBLOCK
    indices = tl.arange(0, XBLOCK)[:]
    # Create a mask of True values with size XBLOCK
    mask = tl.full([XBLOCK], True, tl.int1)
    
    # Load a single element from the input pointer
    input_value = tl.load(in_ptr0 + (0))
    # Broadcast the loaded value to a vector of size XBLOCK
    broadcasted_input = tl.broadcast_to(input_value, [XBLOCK])
    # Create a vector of ones with size 1 and type int64
    ones_vector = tl.full([1], 1, tl.int64)
    # Add the broadcasted input to the ones vector
    result_vector = broadcasted_input + ones_vector
    
    # Store the result in the output pointer with an offset of 0
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), result_vector, None)