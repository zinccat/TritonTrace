# From: 49_Mamba2ReturnFinalState

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_1poi_fused_clone_1(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    xnumel = 64
    y_offset = tl.program_id(1) * YBLOCK
    y_index = y_offset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    x_coord = x_index
    y_mod_128 = y_index % 128
    y_div_128 = y_index // 128
    y_div_16_mod_8 = (y_index // 16) % 8
    y_div_128_mod_2 = (y_index // 128) % 2
    y_div_256 = y_index // 256
    y_full_index = y_index
    tmp0 = tl.load(in_ptr0 + (y_mod_128 + 128 * x_coord + 8192 * y_div_128), x_mask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (63 + 64 * y_div_128_mod_2 + 128 * y_div_16_mod_8 + 1024 * y_div_256), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x_coord + 64 * y_div_128_mod_2 + 128 * y_div_16_mod_8 + 1024 * y_div_256), x_mask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp4 = tl.math.exp(tmp3)
    tmp5 = tmp0 * tmp4
    tl.store(out_ptr0 + (x_coord + 64 * y_full_index), tmp5, x_mask)