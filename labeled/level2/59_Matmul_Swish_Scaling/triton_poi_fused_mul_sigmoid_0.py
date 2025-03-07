# From: 59_Matmul_Swish_Scaling

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_mul_sigmoid_0(input_ptr, output_ptr, num_elements, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    indices = block_indices
    input_values = tl.load(input_ptr + (indices), None)
    sigmoid_values = tl.sigmoid(input_values)
    elementwise_product = input_values * sigmoid_values
    scaling_factor = 2.0
    scaled_result = elementwise_product * scaling_factor
    tl.store(output_ptr + (indices), scaled_result, None)