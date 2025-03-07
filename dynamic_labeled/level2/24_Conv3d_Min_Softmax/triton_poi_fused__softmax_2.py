# From: 24_Conv3d_Min_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax_2(in_out_ptr0, in_ptr0, in_ptr1, kernel_size0, kernel_size1, kernel_size2, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    linear_index = indices
    kernel_index0 = indices % kernel_size0
    linear_index_div_k1 = indices // kernel_size1
    loaded_value = tl.load(in_out_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    loaded_input0 = tl.load(in_ptr0 + (kernel_index0 + 4*linear_index_div_k1 + linear_index_div_k1*kernel_size2*kernel_size2 + ((-4)*kernel_size2*linear_index_div_k1)), mask, eviction_policy='evict_last')
    loaded_input1 = tl.load(in_ptr1 + (kernel_index0 + 4*linear_index_div_k1 + linear_index_div_k1*kernel_size2*kernel_size2 + ((-4)*kernel_size2*linear_index_div_k1)), mask, eviction_policy='evict_last')
    subtracted_value = loaded_value - loaded_input0
    exp_value = tl.math.exp(subtracted_value)
    softmax_result = exp_value / loaded_input1
    tl.store(in_out_ptr0 + (linear_index), softmax_result, mask)