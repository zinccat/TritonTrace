# From: 63_Gemm_ReLU_Divide

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_div_relu_threshold_backward_0(input_ptr0, input_ptr1, output_ptr0, output_ptr1, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    element_index = index
    block_index = index % 512
    input_value0 = tl.load(input_ptr0 + (element_index), None)
    input_value1 = tl.load(input_ptr1 + (block_index), None, eviction_policy='evict_last')
    sum_values = input_value0 + input_value1
    zero_value = tl.full([1], 0, tl.int32)
    max_value = triton_helpers.maximum(zero_value, sum_values)
    half_value = 0.5
    relu_result = max_value * half_value
    zero_threshold = 0.0
    threshold_condition = max_value <= zero_threshold
    tl.store(output_ptr0 + (element_index), relu_result, None)
    tl.store(output_ptr1 + (element_index), threshold_condition, None)