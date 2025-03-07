# From: 36_RMSNorm_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_div_mean_pow_sqrt_1(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    linear_index = xindex
    x_mod_ks0 = xindex % ks0
    x_div_ks1 = xindex // ks1
    
    input_data0 = tl.load(in_ptr0 + (linear_index), xmask, eviction_policy='evict_last')
    input_data1 = tl.load(in_ptr1 + (x_mod_ks0 + ks0 * x_div_ks1), xmask, eviction_policy='evict_last')
    
    divisor = ks2
    divisor_float = divisor.to(tl.float32)
    
    normalized_data = input_data1 / divisor_float
    epsilon = 1e-05
    adjusted_data = normalized_data + epsilon
    
    sqrt_data = tl.extra.cuda.libdevice.sqrt(adjusted_data)
    
    result_data = input_data0 / sqrt_data
    
    tl.store(out_ptr0 + (linear_index), result_data, xmask)