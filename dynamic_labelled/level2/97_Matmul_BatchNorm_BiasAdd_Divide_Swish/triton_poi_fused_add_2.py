# From: 97_Matmul_BatchNorm_BiasAdd_Divide_Swish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_2(in_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    # Calculate the starting offset for the current program ID
    start_offset = tl.program_id(0) * XBLOCK
    # Generate a range of indices from 0 to XBLOCK
    indices = tl.arange(0, XBLOCK)[:]
    # Create a mask of ones with size XBLOCK
    mask = tl.full([XBLOCK], True, tl.int1)
    
    # Load a value from the input pointer
    input_value = tl.load(in_ptr0 + (0))
    # Broadcast the loaded value to a vector of size XBLOCK
    broadcasted_value = tl.broadcast_to(input_value, [XBLOCK])
    # Create a scalar value of 1 with type int64
    scalar_one = tl.full([1], 1, tl.int64)
    # Add the scalar value to the broadcasted vector
    result = broadcasted_value + scalar_one
    
    # Store the result in the output pointer with an offset of zero
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), result, None)