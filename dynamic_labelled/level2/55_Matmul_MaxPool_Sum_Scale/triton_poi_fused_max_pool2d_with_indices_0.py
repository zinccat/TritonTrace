# From: 55_Matmul_MaxPool_Sum_Scale

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_0poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # Calculate the starting index for the current program
    start_index = tl.program_id(0) * XBLOCK
    # Generate a range of indices from start_index
    indices = start_index + tl.arange(0, XBLOCK)[:]
    # Create a mask to ensure indices are within bounds
    valid_mask = indices < xnumel
    
    # Calculate positions for loading data
    pos0 = (indices % 2)
    pos1 = indices // 2
    pos2 = indices
    
    # Load data from input pointer with eviction policy
    data0 = tl.load(in_ptr0 + (2 * pos0 + 5 * pos1), valid_mask, eviction_policy='evict_last')
    data1 = tl.load(in_ptr0 + (1 + 2 * pos0 + 5 * pos1), valid_mask, eviction_policy='evict_last')
    
    # Compare data1 and data0
    is_greater = data1 > data0
    
    # Create tensors for storing comparison results
    true_tensor = tl.full([1], 1, tl.int8)
    false_tensor = tl.full([1], 0, tl.int8)
    
    # Store the result of the comparison
    comparison_result = tl.where(is_greater, true_tensor, false_tensor)
    
    # Compute the maximum of data1 and data0
    triton_helpers.maximum(data1, data0)
    
    # Store the comparison result in the output pointer
    tl.store(out_ptr0 + (pos2), comparison_result, valid_mask)