# From: 62_Matmul_GroupNorm_LeakyReLU_Sum

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_add_leaky_relu_native_group_norm_1(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    # Indices and offsets
    x2 = xindex
    x0 = xindex % 256
    
    # Load inputs
    input0 = tl.load(in_ptr0 + (x2), None)
    input1 = tl.load(in_ptr1 + ((x2 // 32)), None, eviction_policy='evict_last')
    input2 = tl.load(in_ptr2 + ((x2 // 32)), None, eviction_policy='evict_last')
    input3 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    input4 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    
    # Compute intermediate values
    normalized = input0 - input1
    scaled = normalized * input2
    weighted = scaled * input3
    biased = weighted + input4
    
    # Leaky ReLU
    zero = 0.0
    leaky_slope = 0.01
    positive = biased > zero
    leaky = biased * leaky_slope
    leaky_relu = tl.where(positive, biased, leaky)
    
    # Summation
    doubled = leaky_relu + leaky_relu
    
    # Store result
    tl.store(in_out_ptr0 + (x2), doubled, None)