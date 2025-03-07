# From: 36_ConvTranspose2d_Min_Sum_GELU_Add

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_add_gelu_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    index_mod_64 = block_indices % 64
    index_div_1024 = block_indices // 1024
    index_div_64_mod_16 = (block_indices // 64) % 16
    linear_index = block_indices
    
    input_value0 = tl.load(in_ptr0 + (index_mod_64 + (64 * index_div_1024)), None, eviction_policy='evict_last')
    input_value1 = tl.load(in_ptr1 + (index_div_64_mod_16), None, eviction_policy='evict_last')
    
    half = 0.5
    scaled_input0 = input_value0 * half
    
    sqrt_half = 0.7071067811865476
    scaled_input0_sqrt_half = input_value0 * sqrt_half
    
    erf_result = tl.extra.cuda.libdevice.erf(scaled_input0_sqrt_half)
    
    one = 1.0
    erf_plus_one = erf_result + one
    
    gelu_result = scaled_input0 * erf_plus_one
    
    fused_result = gelu_result + input_value1
    
    tl.store(out_ptr0 + (linear_index), fused_result, None)