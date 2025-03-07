# From: 18_SqueezeNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_10poi_fused_cat_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 373248
    program_id_offset = tl.program_id(0) * XBLOCK
    index_within_block = program_id_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = index_within_block < xnumel
    index_mod_128 = index_within_block % 128
    index_div_128 = index_within_block // 128
    original_index = index_within_block

    zero_value = tl.full([1], 0, tl.int64)
    sixty_four_value = tl.full([1], 64, tl.int64)
    is_less_than_sixty_four = index_mod_128 < sixty_four_value

    load_value_0 = tl.load(in_ptr0 + (64 * index_div_128 + index_mod_128), is_less_than_sixty_four & valid_mask, eviction_policy='evict_last', other=0.0)
    load_value_1 = tl.load(in_ptr1 + index_mod_128, is_less_than_sixty_four & valid_mask, eviction_policy='evict_last', other=0.0)
    sum_values = load_value_0 + load_value_1

    zero_int32 = tl.full([1], 0, tl.int32)
    max_value_0 = triton_helpers.maximum(zero_int32, sum_values)
    zero_float = tl.full(max_value_0.shape, 0.0, max_value_0.dtype)
    selected_value_0 = tl.where(is_less_than_sixty_four, max_value_0, zero_float)

    is_greater_equal_sixty_four = index_mod_128 >= sixty_four_value

    load_value_2 = tl.load(in_ptr2 + (64 * index_div_128 + ((-64) + index_mod_128)), is_greater_equal_sixty_four & valid_mask, eviction_policy='evict_last', other=0.0)
    load_value_3 = tl.load(in_ptr3 + ((-64) + index_mod_128), is_greater_equal_sixty_four & valid_mask, eviction_policy='evict_last', other=0.0)
    sum_values_2 = load_value_2 + load_value_3

    max_value_1 = triton_helpers.maximum(zero_int32, sum_values_2)
    selected_value_1 = tl.where(is_greater_equal_sixty_four, max_value_1, zero_float)

    final_selected_value = tl.where(is_less_than_sixty_four, selected_value_0, selected_value_1)

    tl.store(out_ptr0 + original_index, final_selected_value, valid_mask)