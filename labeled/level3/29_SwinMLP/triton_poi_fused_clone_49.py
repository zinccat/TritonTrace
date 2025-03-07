# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_49poi_fused_clone_49(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x_mod_32 = xindex % 32
    x_div_32_mod_49 = (xindex // 32) % 49
    x_div_1568_mod_24 = (xindex // 1568) % 24
    x_div_37632 = xindex // 37632
    x_full_index = xindex

    input_value0 = tl.load(in_ptr0 + (x_mod_32 + 32 * x_div_1568_mod_24 + 768 * x_div_32_mod_49 + 37632 * x_div_37632), xmask)
    input_value1 = tl.load(in_ptr1 + (x_mod_32 + 32 * x_div_1568_mod_24), xmask, eviction_policy='evict_last')
    input_value2 = tl.load(in_ptr2 + (x_mod_32 + 32 * x_div_1568_mod_24), xmask, eviction_policy='evict_last')

    multiplied_values = input_value0 * input_value1
    result_value = multiplied_values + input_value2

    tl.store(out_ptr0 + (x_full_index), result_value, xmask)