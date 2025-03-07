# From: 90_Conv3d_LeakyReLU_Sum_Clamp_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_add_clamp_convolution_gelu_leaky_relu_0(
    in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    # Calculate indices
    input_index = xindex
    channel_index = (xindex // 12600) % 16
    
    # Load data
    output_accumulator = tl.load(in_out_ptr0 + (input_index), None)
    input_feature_map = tl.load(in_ptr0 + (channel_index), None, eviction_policy='evict_last')
    bias = tl.load(in_ptr1 + (channel_index), None, eviction_policy='evict_last')
    
    # Perform addition
    added_result = output_accumulator + input_feature_map
    
    # Apply Leaky ReLU
    zero = 0.0
    leaky_relu_slope = 0.2
    is_positive = added_result > zero
    leaky_relu_result = tl.where(is_positive, added_result, added_result * leaky_relu_slope)
    
    # Add bias
    biased_result = leaky_relu_result + bias
    
    # Clamp result
    clamp_min = -1.0
    clamp_max = 1.0
    clamped_result = triton_helpers.maximum(biased_result, clamp_min)
    clamped_result = triton_helpers.minimum(clamped_result, clamp_max)
    
    # Apply GELU
    gelu_coefficient = 0.5
    gelu_sqrt_2_over_sqrt_pi = 0.7071067811865476
    gelu_clamped = clamped_result * gelu_coefficient
    gelu_erf_input = clamped_result * gelu_sqrt_2_over_sqrt_pi
    erf_result = tl.extra.cuda.libdevice.erf(gelu_erf_input)
    gelu_result = gelu_clamped * (erf_result + clamp_max)
    
    # Store results
    tl.store(in_out_ptr0 + (input_index), added_result, None)
    tl.store(out_ptr0 + (input_index), gelu_result, None)