# From: 80_Gemm_Max_Subtract_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_div_gelu_backward_neg_scatter_zeros_1(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    base_indices = indices
    input_values0 = tl.load(input_ptr0 + (base_indices), mask)
    input_values1 = tl.load(input_ptr1 + (base_indices), mask)
    input_values2 = tl.load(input_ptr2 + (base_indices), mask)
    
    tl.device_assert(((0 <= input_values0) & (input_values0 < 1024)) | ~mask, 
                     "index out of bounds: 0 <= input_values0 < 1024")
    
    product = input_values1 * input_values2
    neg_product = -product
    scalar = 1.0
    neg_product_scaled = neg_product * scalar
    result = product + neg_product_scaled
    
    tl.store(output_ptr0 + (input_values0 + 1024 * base_indices), result, mask)