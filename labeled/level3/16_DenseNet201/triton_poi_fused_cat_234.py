# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_234poi_fused_cat_234(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 486080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    block_index = (xindex // 49) % 992
    within_block_index = xindex % 49
    batch_index = xindex // 48608
    linear_index = xindex

    block_index_copy = block_index
    zero_value = tl.full([1], 0, tl.int64)
    max_index_1 = tl.full([1], 896, tl.int64)
    condition_1 = block_index_copy < max_index_1
    load_1 = tl.load(in_ptr0 + (within_block_index + 49 * block_index_copy + 43904 * batch_index), condition_1 & xmask, other=0.0)

    max_index_2 = tl.full([1], 928, tl.int64)
    condition_2 = block_index_copy >= max_index_1
    condition_3 = block_index_copy < max_index_2
    condition_4 = condition_2 & condition_3
    load_2 = tl.load(in_ptr1 + (within_block_index + 49 * ((-896) + block_index_copy) + 1568 * batch_index), condition_4 & xmask, other=0.0)

    max_index_3 = tl.full([1], 960, tl.int64)
    condition_5 = block_index_copy >= max_index_2
    condition_6 = block_index_copy < max_index_3
    condition_7 = condition_5 & condition_6
    load_3 = tl.load(in_ptr2 + (within_block_index + 49 * ((-928) + block_index_copy) + 1568 * batch_index), condition_7 & xmask, other=0.0)

    max_index_4 = tl.full([1], 992, tl.int64)
    condition_8 = block_index_copy >= max_index_3
    load_4 = tl.load(in_ptr3 + (within_block_index + 49 * ((-960) + block_index_copy) + 1568 * batch_index), condition_8 & xmask, other=0.0)

    select_1 = tl.where(condition_7, load_3, load_4)
    select_2 = tl.where(condition_4, load_2, select_1)
    select_3 = tl.where(condition_1, load_1, select_2)

    tl.store(out_ptr0 + (linear_index), select_3, xmask)