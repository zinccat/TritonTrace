# From: 98_Matmul_AvgPool_GELU_Scale_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool2d_backward_3poi_fused_avg_pool2d_backward_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # Calculate the starting index for the current program
    start_index = tl.program_id(0) * XBLOCK
    # Generate a range of indices from start_index
    indices = start_index + tl.arange(0, XBLOCK)[:]
    # Create a mask to ensure indices are within bounds
    valid_mask = indices < xnumel
    
    # Calculate various components for indexing
    index_mod_256 = indices % 256
    index_div_256 = indices // 256
    full_index = indices
    
    # Load data from input pointer with calculated offset and mask
    offset_component = 64 * index_div_256
    inner_offset = (((0) * ((0) >= (index_mod_256 // 4)) + (index_mod_256 // 4) * ((index_mod_256 // 4) > (0))))
    range_limit = ((((0) * ((0) >= (index_mod_256 // 4)) + (index_mod_256 // 4) * ((index_mod_256 // 4) > (0)))) <= ((-1) + ((64) * ((64) <= (1 + (index_mod_256 // 4))) + (1 + (index_mod_256 // 4)) * ((1 + (index_mod_256 // 4)) < (64)))))
    final_offset = ((-1) + ((64) * ((64) <= (1 + (index_mod_256 // 4))) + (1 + (index_mod_256 // 4)) * ((1 + (index_mod_256 // 4)) < (64))))
    valid_range = (((-1) + ((64) * ((64) <= (1 + (index_mod_256 // 4))) + (1 + (index_mod_256 // 4)) * ((1 + (index_mod_256 // 4)) < (64)))) < (((0) * ((0) >= (index_mod_256 // 4)) + (index_mod_256 // 4) * ((index_mod_256 // 4) > (0)))))
    
    loaded_data = tl.load(in_ptr0 + (offset_component + (inner_offset * range_limit * final_offset * valid_range)), valid_mask, eviction_policy='evict_last')
    
    # Perform division
    divided_data = loaded_data / 4
    
    # Create temporary tensors
    zero_tensor = tl.full([1], 0, tl.int32)
    one_tensor = tl.full([1], 1, tl.int32)
    
    # Create masks for conditions
    zero_less_one_mask = zero_tensor < one_tensor
    inner_offset_mask = ((0) * ((0) >= (index_mod_256 // 4)) + (index_mod_256 // 4) * ((index_mod_256 // 4) > (0)))
    range_limit_mask = ((64) * ((64) <= (1 + (index_mod_256 // 4))) + (1 + (index_mod_256 // 4)) * ((1 + (index_mod_256 // 4)) < (64)))
    valid_inner_offset_mask = inner_offset_mask < range_limit_mask
    combined_mask = zero_less_one_mask & valid_inner_offset_mask
    
    # Use where to select between divided data and zero
    result_data = tl.where(combined_mask, divided_data, 0.0)
    
    # Store the result back to the output pointer
    tl.store(out_ptr0 + (full_index), result_data, valid_mask)