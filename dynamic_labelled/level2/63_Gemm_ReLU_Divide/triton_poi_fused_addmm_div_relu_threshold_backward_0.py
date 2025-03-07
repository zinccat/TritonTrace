# From: 63_Gemm_ReLU_Divide

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_div_relu_threshold_backward_0poi_fused_addmm_div_relu_threshold_backward_0(input_ptr0, input_ptr1, output_ptr0, output_ptr1, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements
    global_indices = block_indices
    local_indices = block_indices % 512
    input_data0 = tl.load(input_ptr0 + (global_indices), mask)
    input_data1 = tl.load(input_ptr1 + (local_indices), mask, eviction_policy='evict_last')
    sum_data = input_data0 + input_data1
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, sum_data)
    half_tensor = 0.5
    scaled_output = relu_output * half_tensor
    zero_comparison = 0.0
    relu_mask = relu_output <= zero_comparison
    tl.store(output_ptr0 + (global_indices), scaled_output, mask)
    tl.store(output_ptr1 + (global_indices), relu_mask, mask)