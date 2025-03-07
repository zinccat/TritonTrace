# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_25poi_fused_cat_25(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 376320
    program_id = tl.program_id(0)
    xoffset = program_id * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x_mod_192 = xindex % 192
    x_div_192_mod_14 = (xindex // 192) % 14
    x_div_2688 = xindex // 2688
    x_div_192 = xindex // 192

    tmp0 = tl.load(in_ptr0 + (192 + x_mod_192 + 384 * x_div_192_mod_14 + 10752 * x_div_2688), xmask)
    tl.store(out_ptr0 + (x_mod_192 + 768 * x_div_192), tmp0, xmask)