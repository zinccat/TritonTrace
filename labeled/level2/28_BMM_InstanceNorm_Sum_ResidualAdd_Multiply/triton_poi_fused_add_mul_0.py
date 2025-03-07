# From: 28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_add_mul_0(input_ptr0, input_ptr1, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    indices = block_indices
    input_value0 = tl.load(input_ptr0 + (indices), None)
    input_value1 = tl.load(input_ptr1 + (indices), None)
    sum_result = input_value0 + input_value1
    multiply_result = sum_result * input_value1
    tl.store(output_ptr0 + (indices), multiply_result, None)