# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_div_76poi_fused_clone_div_76(input_ptr0, input_ptr1, output_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x_mod_32 = xindex % 32
    x_div_32_mod_49 = (xindex // 32) % 49
    x_div_1568_mod_24 = (xindex // 1568) % 24
    x_div_37632 = xindex // 37632
    x_div_32_mod_1176 = (xindex // 32) % 1176
    x_full_index = xindex

    input_value0 = tl.load(input_ptr0 + (x_mod_32 + 32 * x_div_1568_mod_24 + 2304 * x_div_32_mod_49 + 112896 * x_div_37632), xmask)
    input_value1 = tl.load(input_ptr1 + (x_div_32_mod_1176 + 1184 * x_div_37632), xmask, eviction_policy='evict_last')
    epsilon = 1e-12
    max_value = triton_helpers.maximum(input_value1, epsilon)
    result_value = input_value0 / max_value

    tl.store(output_ptr0 + (x_full_index), result_value, xmask)