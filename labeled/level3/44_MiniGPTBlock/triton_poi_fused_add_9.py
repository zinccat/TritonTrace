# From: 44_MiniGPTBlock

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_9poi_fused_add_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = x_index
    x0 = (x_index % 768)
    
    input_value0 = tl.load(in_ptr0 + (x2), None)
    input_value1 = tl.load(in_ptr1 + (x2), None)
    output_value = tl.load(in_out_ptr0 + (x2), None)
    input_value2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    
    sum_input_values = input_value0 + input_value1
    sum_output_input2 = output_value + input_value2
    final_result = sum_input_values + sum_output_input2
    
    tl.store(in_out_ptr0 + (x2), final_result, None)