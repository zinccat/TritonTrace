# From: 63_Gemm_ReLU_Divide

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_div_relu_threshold_backward_0(input_ptr0, input_ptr1, output_ptr0, output_ptr1, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    element_indices = indices
    block_indices = element_indices % 512
    loaded_input0 = tl.load(input_ptr0 + (element_indices), mask)
    loaded_input1 = tl.load(input_ptr1 + (block_indices), mask, eviction_policy='evict_last')
    added_values = loaded_input0 + loaded_input1
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_applied = triton_helpers.maximum(zero_tensor, added_values)
    half_tensor = 0.5
    scaled_values = relu_applied * half_tensor
    zero_comparison = 0.0
    relu_mask = relu_applied <= zero_comparison
    tl.store(output_ptr0 + (element_indices), scaled_values, mask)
    tl.store(output_ptr1 + (element_indices), relu_mask, mask)