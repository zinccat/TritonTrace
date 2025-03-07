# From: 48_Mamba2ReturnY

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mul_4poi_fused_mul_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    x_index_mod_4096 = x_index % 4096
    x_index_div_64_mod_64 = (x_index // 64) % 64
    x_index_div_4096_mod_8 = (x_index // 4096) % 8
    x_index_div_32768_mod_2 = (x_index // 32768) % 2
    x_index_div_65536 = x_index // 65536
    x_index_mod_64 = x_index % 64
    x_index_div_32768 = x_index // 32768
    
    tmp0 = tl.load(in_ptr0 + x_index, None)
    tmp1 = tl.load(in_ptr1 + x_index_mod_4096, None, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tl.load(in_ptr2 + (x_index_div_64_mod_64 + 64 * x_index_div_32768_mod_2 + 128 * x_index_div_4096_mod_8 + 1024 * x_index_div_65536), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x_index_mod_64 + 64 * x_index_div_32768_mod_2 + 128 * x_index_div_4096_mod_8 + 1024 * x_index_div_65536), None, eviction_policy='evict_last')
    
    tmp4 = tmp2 - tmp3
    tmp5 = float("-inf")
    tmp6 = tl.where(tmp1, tmp5, tmp4)
    tmp7 = tl.math.exp(tmp6)
    tmp8 = tmp0 * tmp7
    
    tl.store(out_ptr0 + (x_index_mod_64 + 64 * x_index_div_4096_mod_8 + 512 * x_index_div_64_mod_64 + 32768 * x_index_div_32768), tmp8, None)