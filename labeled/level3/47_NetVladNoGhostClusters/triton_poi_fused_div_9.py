# From: 47_NetVladNoGhostClusters

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_9poi_fused_div_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    x_mod_16384 = xindex % 16384
    x_div_16384 = xindex // 16384
    x_full_index = xindex
    
    input_val0 = tl.load(in_ptr0 + (512 * (x_mod_16384 % 32) + 16384 * x_div_16384 + (x_mod_16384 // 32)), None, eviction_policy='evict_last')
    input_val1 = tl.load(in_ptr1 + (32 * x_div_16384 + (x_mod_16384 % 32)), None)
    input_val2 = tl.load(in_ptr2 + (x_mod_16384), None, eviction_policy='evict_last')
    input_val3 = tl.load(in_ptr3 + (32 * x_div_16384 + (x_mod_16384 % 32)), None)
    input_val4 = tl.load(in_ptr4 + (x_div_16384), None, eviction_policy='evict_last')
    
    intermediate_val1 = input_val1 * input_val2
    intermediate_val2 = input_val0 - intermediate_val1
    epsilon = 1e-12
    max_val1 = triton_helpers.maximum(input_val3, epsilon)
    result_val1 = intermediate_val2 / max_val1
    max_val2 = triton_helpers.maximum(input_val4, epsilon)
    final_result = result_val1 / max_val2
    
    tl.store(out_ptr0 + (x_full_index), final_result, None)