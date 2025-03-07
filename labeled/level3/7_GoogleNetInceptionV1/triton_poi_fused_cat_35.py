# From: 7_GoogleNetInceptionV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_35poi_fused_cat_35(
    input_ptr_0, input_ptr_1, input_ptr_2, input_ptr_3, input_ptr_4, input_ptr_5, input_ptr_6, input_ptr_7,
    output_ptr_0, num_elements, BLOCK_SIZE : tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    index_mod_512 = block_indices % 512
    index_div_512 = block_indices // 512
    global_index = block_indices
    
    mod_512 = index_mod_512
    full_zero = tl.full([1], 0, tl.int64)
    full_192 = tl.full([1], 192, tl.int64)
    condition_192 = mod_512 < full_192
    
    load_0 = tl.load(input_ptr_0 + (192 * index_div_512 + mod_512), condition_192, eviction_policy='evict_last', other=0.0)
    load_1 = tl.load(input_ptr_1 + mod_512, condition_192, eviction_policy='evict_last', other=0.0)
    sum_192 = load_0 + load_1
    zero_like_sum_192 = tl.full(sum_192.shape, 0.0, sum_192.dtype)
    result_192 = tl.where(condition_192, sum_192, zero_like_sum_192)
    
    condition_400 = (mod_512 >= full_192) & (mod_512 < tl.full([1], 400, tl.int64))
    load_2 = tl.load(input_ptr_2 + (208 * index_div_512 + (mod_512 - 192)), condition_400, eviction_policy='evict_last', other=0.0)
    load_3 = tl.load(input_ptr_3 + (mod_512 - 192), condition_400, eviction_policy='evict_last', other=0.0)
    sum_400 = load_2 + load_3
    zero_like_sum_400 = tl.full(sum_400.shape, 0.0, sum_400.dtype)
    result_400 = tl.where(condition_400, sum_400, zero_like_sum_400)
    
    condition_448 = (mod_512 >= tl.full([1], 400, tl.int64)) & (mod_512 < tl.full([1], 448, tl.int64))
    load_4 = tl.load(input_ptr_4 + (48 * index_div_512 + (mod_512 - 400)), condition_448, eviction_policy='evict_last', other=0.0)
    load_5 = tl.load(input_ptr_5 + (mod_512 - 400), condition_448, eviction_policy='evict_last', other=0.0)
    sum_448 = load_4 + load_5
    zero_like_sum_448 = tl.full(sum_448.shape, 0.0, sum_448.dtype)
    result_448 = tl.where(condition_448, sum_448, zero_like_sum_448)
    
    condition_512 = mod_512 >= tl.full([1], 448, tl.int64)
    load_6 = tl.load(input_ptr_6 + (64 * index_div_512 + (mod_512 - 448)), condition_512, eviction_policy='evict_last', other=0.0)
    load_7 = tl.load(input_ptr_7 + (mod_512 - 448), condition_512, eviction_policy='evict_last', other=0.0)
    sum_512 = load_6 + load_7
    zero_like_sum_512 = tl.full(sum_512.shape, 0.0, sum_512.dtype)
    result_512 = tl.where(condition_512, sum_512, zero_like_sum_512)
    
    final_result = tl.where(condition_448, result_448, result_512)
    final_result = tl.where(condition_400, result_400, final_result)
    final_result = tl.where(condition_192, result_192, final_result)
    
    tl.store(output_ptr_0 + global_index, final_result, None)