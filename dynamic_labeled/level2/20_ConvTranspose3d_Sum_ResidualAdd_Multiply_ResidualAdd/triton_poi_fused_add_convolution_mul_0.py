# From: 20_ConvTranspose3d_Sum_ResidualAdd_Multiply_ResidualAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_convolution_mul_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, kernel_size, num_elements, XBLOCK : tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = x_index
    x1 = ((x_index // kernel_size) % 64)
    
    # Load data from pointers
    input_output_value = tl.load(in_out_ptr0 + (x3), None, eviction_policy='evict_last')
    input_value_0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    input_value_1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    
    # Perform computations
    sum_0 = input_output_value + input_value_0
    sum_1 = sum_0 + input_value_1
    residual_add_1 = sum_1 + sum_0
    multiply_result = residual_add_1 * sum_0
    residual_add_2 = multiply_result + sum_0
    
    # Store results
    tl.store(in_out_ptr0 + (x3), sum_0, None)
    tl.store(out_ptr0 + (x3), residual_add_2, None)