# From: 46_NetVladWithGhostClusters

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_9poi_fused_div_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    x_mod_16384 = x_index % 16384
    x_div_16384 = x_index // 16384
    x_full_index = x_index
    
    tmp0 = tl.load(in_ptr0 + (512 * (x_mod_16384 % 32) + 16384 * x_div_16384 + (x_mod_16384 // 32)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (32 * x_div_16384 + (x_mod_16384 % 32)), None)
    tmp2 = tl.load(in_ptr2 + (x_mod_16384), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (32 * x_div_16384 + (x_mod_16384 % 32)), None)
    tmp9 = tl.load(in_ptr4 + (x_div_16384), None, eviction_policy='evict_last')
    
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 - tmp3
    epsilon = 1e-12
    tmp7 = triton_helpers.maximum(tmp5, epsilon)
    tmp8 = tmp4 / tmp7
    tmp10 = triton_helpers.maximum(tmp9, epsilon)
    tmp11 = tmp8 / tmp10
    
    tl.store(out_ptr0 + (x_full_index), tmp11, None)