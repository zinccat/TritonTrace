# From: 76_Gemm_Add_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_relu_threshold_backward_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    
    # Load data from input pointers
    in_out_data = tl.load(in_out_ptr0 + (x2), xmask)
    in_data = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    
    # Perform addition
    added_data = in_out_data + in_data
    
    # Create a tensor filled with zeros
    zero_tensor = tl.full([1], 0, tl.int32)
    
    # Apply ReLU operation
    relu_result = triton_helpers.maximum(zero_tensor, added_data)
    
    # Threshold value
    threshold_value = 0.0
    
    # Create a mask for values less than or equal to the threshold
    threshold_mask = relu_result <= threshold_value
    
    # Store results back to the output pointers
    tl.store(in_out_ptr0 + (x2), relu_result, xmask)
    tl.store(out_ptr0 + (x2), threshold_mask, xmask)