# From: 51_Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_sum_0per_fused_sum_0(input_ptr, output_ptr, num_elements_x, num_elements_r):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 1024
    
    # Calculate the offset for the current program ID
    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    
    # Initialize a block of boolean values set to True
    r_index = tl.arange(0, RBLOCK)[:]
    
    # Load data from input pointer with calculated indices
    r1 = r_index
    x0 = x_index
    temp_data = tl.load(input_ptr + (r1 + 1024 * x0), None)
    
    # Broadcast the loaded data across the block
    broadcast_data = tl.broadcast_to(temp_data, [RBLOCK])
    
    # Sum the broadcasted data and promote to tensor
    summed_data = triton_helpers.promote_to_tensor(tl.sum(broadcast_data, 0))
    
    # Store the result in the output pointer
    tl.store(output_ptr + (x0), summed_data, None)