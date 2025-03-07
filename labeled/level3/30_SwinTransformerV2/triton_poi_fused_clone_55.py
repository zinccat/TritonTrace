# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_55poi_fused_clone_55(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    xnumel = 49
    y_offset = tl.program_id(1) * YBLOCK
    y_index = y_offset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    x_coord = x_index
    y_block_index = y_index // 384
    y_within_block = y_index % 384
    y_quarter_index = y_index // 32
    y_full_index = y_index
    input_data0 = tl.load(in_ptr0 + (384 + y_within_block + 1152 * x_coord + 56448 * y_block_index), x_mask, eviction_policy='evict_last')
    input_data1 = tl.load(in_ptr1 + (x_coord + 49 * y_quarter_index), x_mask, eviction_policy='evict_last')
    epsilon = 1e-12
    max_value = triton_helpers.maximum(input_data1, epsilon)
    result = input_data0 / max_value
    tl.store(out_ptr0 + (x_coord + 49 * y_full_index), result, x_mask)