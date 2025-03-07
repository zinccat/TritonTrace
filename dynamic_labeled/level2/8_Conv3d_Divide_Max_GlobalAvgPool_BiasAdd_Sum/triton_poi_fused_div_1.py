# From: 8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_1(input_ptr, output_ptr, kernel_size_z, kernel_size_y, kernel_size_x, num_elements, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = block_indices < num_elements

    # Calculate indices for accessing the input tensor
    depth_index = block_indices // kernel_size_x
    flat_index = block_indices

    # Load input data with eviction policy
    input_data = tl.load(input_ptr + (depth_index), valid_mask, eviction_policy='evict_last')

    # Compute divisor based on kernel dimensions
    divisor = (-1) + ((-1) * (kernel_size_x // 2) * (kernel_size_x // 2)) + 2 * (kernel_size_x // 2) + (kernel_size_x // 2) * (kernel_size_x // 2) * (kernel_size_y // 2) + ((-2) * (kernel_size_y // 2) * (kernel_size_x // 2)) + (kernel_size_y // 2)

    # Convert divisor to float32
    divisor_float = divisor.to(tl.float32)

    # Perform division
    result = input_data / divisor_float

    # Store the result
    tl.store(output_ptr + (flat_index), result, valid_mask)