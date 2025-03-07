# From: 49_ConvTranspose3d_Softmax_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax__softmax_backward_data_2poi_fused__softmax__softmax_backward_data_2(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, kernel_size0, kernel_size1, kernel_size2, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    # Calculate indices
    batch_index = xindex
    kernel_index0 = xindex % kernel_size0
    kernel_index2 = xindex // kernel_size1
    
    # Load data
    output_value = tl.load(in_out_ptr0 + (batch_index), None, eviction_policy='evict_last')
    input_value0 = tl.load(in_ptr0 + (kernel_index0 + 8192 * kernel_size2 * kernel_index2), None, eviction_policy='evict_last')
    input_value1 = tl.load(in_ptr1 + (kernel_index0 + 8192 * kernel_size2 * kernel_index2), None, eviction_policy='evict_last')
    input_value2 = tl.load(in_ptr2 + (kernel_index0 + 8192 * kernel_size2 * kernel_index2), None, eviction_policy='evict_last')
    input_value3 = tl.load(in_ptr3 + (batch_index), None, eviction_policy='evict_last')
    
    # Compute intermediate values
    diff = output_value - input_value0
    exp_diff = tl.math.exp(diff)
    softmax_output = exp_diff / input_value1
    neg_softmax_output = -softmax_output
    
    # Compute final result
    fused_result = tl.extra.cuda.libdevice.fma(neg_softmax_output, input_value2, input_value3)
    
    # Store result
    tl.store(in_out_ptr0 + (batch_index), fused_result, None)