# From: 18_SqueezeNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_23poi_fused_cat_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 373248
    program_id_offset = tl.program_id(0) * XBLOCK
    index_within_block = program_id_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = index_within_block < xnumel
    block_offset = index_within_block % 512
    block_index = index_within_block // 512
    global_index = index_within_block
    block_offset_copy = block_offset
    zero_value = tl.full([1], 0, tl.int64)
    half_block_size = tl.full([1], 256, tl.int64)
    is_first_half = block_offset_copy < half_block_size
    load_first_half = tl.load(in_ptr0 + (256 * block_index + block_offset), is_first_half & valid_mask, eviction_policy='evict_last', other=0.0)
    load_second_half = tl.load(in_ptr1 + block_offset, is_first_half & valid_mask, eviction_policy='evict_last', other=0.0)
    sum_first_half = load_first_half + load_second_half
    zero_int32 = tl.full([1], 0, tl.int32)
    max_first_half = triton_helpers.maximum(zero_int32, sum_first_half)
    zero_float32 = tl.full(max_first_half.shape, 0.0, max_first_half.dtype)
    result_first_half = tl.where(is_first_half, max_first_half, zero_float32)
    is_second_half = block_offset_copy >= half_block_size
    full_block_size = tl.full([1], 512, tl.int64)
    load_third_half = tl.load(in_ptr2 + (256 * block_index + (-256 + block_offset)), is_second_half & valid_mask, eviction_policy='evict_last', other=0.0)
    load_fourth_half = tl.load(in_ptr3 + (-256 + block_offset), is_second_half & valid_mask, eviction_policy='evict_last', other=0.0)
    sum_second_half = load_third_half + load_fourth_half
    max_second_half = triton_helpers.maximum(zero_int32, sum_second_half)
    result_second_half = tl.where(is_second_half, max_second_half, zero_float32)
    final_result = tl.where(is_first_half, result_first_half, result_second_half)
    tl.store(out_ptr0 + global_index, final_result, valid_mask)