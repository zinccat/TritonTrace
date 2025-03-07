# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_8poi_fused_clone_8(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    xnumel = 49
    y_offset = tl.program_id(1) * YBLOCK
    y_index = y_offset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    x_coord = x_index
    y_block_96 = y_index // 96
    y_mod_96 = y_index % 96
    y_block_32 = y_index // 32
    y_index_copy = y_index
    input_data_0 = tl.load(in_ptr0 + (96 + y_mod_96 + 288 * x_coord + 14112 * y_block_96), x_mask, eviction_policy='evict_last')
    input_data_1 = tl.load(in_ptr1 + (x_coord + 49 * y_block_32), x_mask, eviction_policy='evict_last')
    epsilon = 1e-12
    max_value = triton_helpers.maximum(input_data_1, epsilon)
    result = input_data_0 / max_value
    tl.store(out_ptr0 + (x_coord + 49 * y_index_copy), result, x_mask)