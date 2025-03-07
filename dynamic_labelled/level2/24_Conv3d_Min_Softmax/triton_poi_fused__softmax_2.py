# From: 24_Conv3d_Min_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax_2poi_fused__softmax_2(in_out_ptr0, in_ptr0, in_ptr1, kernel_size0, kernel_size1, kernel_size2, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    kernel_index0 = index % kernel_size0
    linear_index1 = index // kernel_size1
    
    # Load data from in_out_ptr0
    loaded_out = tl.load(in_out_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    
    # Load data from in_ptr0
    loaded_input0 = tl.load(in_ptr0 + (kernel_index0 + 4*linear_index1 + linear_index1*kernel_size2*kernel_size2 + ((-4)*kernel_size2*linear_index1)), mask, eviction_policy='evict_last')
    
    # Load data from in_ptr1
    loaded_input1 = tl.load(in_ptr1 + (kernel_index0 + 4*linear_index1 + linear_index1*kernel_size2*kernel_size2 + ((-4)*kernel_size2*linear_index1)), mask, eviction_policy='evict_last')
    
    # Compute softmax
    subtracted = loaded_out - loaded_input0
    exponentiated = tl.math.exp(subtracted)
    softmax_result = exponentiated / loaded_input1
    
    # Store the result back to in_out_ptr0
    tl.store(in_out_ptr0 + (linear_index), softmax_result, mask)