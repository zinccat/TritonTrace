# From: 88_Gemm_GroupNorm_Swish_Multiply_Swish

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_mul_native_group_norm_sigmoid_1(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    # Indices and pointers
    x2 = xindex
    x0 = xindex % 1024
    
    # Load inputs
    input0 = tl.load(in_ptr0 + (x2), None)
    input1 = tl.load(in_ptr1 + ((x2 // 64)), None, eviction_policy='evict_last')
    input2 = tl.load(in_ptr2 + ((x2 // 64)), None, eviction_policy='evict_last')
    input3 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    input4 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    input5 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    
    # Computation
    normalized_input = input0 - input1
    scaled_input = normalized_input * input2
    weighted_input = scaled_input * input3
    biased_input = weighted_input + input4
    sigmoid_output = tl.sigmoid(biased_input)
    swish_output = biased_input * sigmoid_output
    final_output = swish_output * input5
    final_sigmoid_output = tl.sigmoid(final_output)
    final_swish_output = final_output * final_sigmoid_output
    
    # Store result
    tl.store(in_out_ptr0 + (x2), final_swish_output, None)