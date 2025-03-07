# From: 97_Matmul_BatchNorm_BiasAdd_Divide_Swish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_div_mul_sigmoid_sigmoid_backward_sum_2(
    input_ptr, output_ptr, num_elements_x, num_elements_r, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 8
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    r_indices = tl.arange(0, RBLOCK)[None, :]
    
    # Load data from input pointer
    r0 = r_indices
    loaded_data = tl.load(input_ptr + (r0), None)
    
    # Broadcast loaded data to match dimensions
    broadcasted_data = tl.broadcast_to(loaded_data, [XBLOCK, RBLOCK])
    
    # Sum across the second dimension
    summed_data = tl.sum(broadcasted_data, 1)[:, None]
    
    # Store the result in the output pointer
    tl.store(output_ptr + (tl.full([XBLOCK, 1], 0, tl.int32)), summed_data, None)