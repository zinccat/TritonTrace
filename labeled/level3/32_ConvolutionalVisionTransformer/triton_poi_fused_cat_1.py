# From: 32_ConvolutionalVisionTransformer

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_1poi_fused_cat_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2560
    block_offset = tl.program_id(0) * XBLOCK
    indices = block_offset + tl.arange(0, XBLOCK)[:]
    mask = indices < xnumel
    block_index = (indices // 128) % 2
    local_index = indices % 128
    block_group = indices // 256
    global_index = indices

    zero_tensor = tl.full([1], 0, tl.int64)
    one_tensor = tl.full([1], 1, tl.int64)
    two_tensor = tl.full([1], 2, tl.int64)

    load_condition_0 = block_index < one_tensor
    value_from_in_ptr0 = tl.load(in_ptr0 + local_index, load_condition_0 & mask, eviction_policy='evict_last', other=0.0)

    load_condition_1 = block_index >= one_tensor
    value_from_in_ptr1 = tl.load(in_ptr1 + (local_index + 128 * block_group), load_condition_1 & mask, eviction_policy='evict_last', other=0.0)

    selected_value = tl.where(load_condition_0, value_from_in_ptr0, value_from_in_ptr1)
    tl.store(out_ptr0 + global_index, selected_value, mask)