# From: 7_GoogleNetInceptionV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_30poi_fused_cat_30(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, input_ptr7,
    output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr
):
    num_elements = 3763200
    block_offset = tl.program_id(0) * BLOCK_SIZE
    indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = indices < num_elements
    x_mod_480 = indices % 480
    x_div_480 = indices // 480
    x_index = indices

    zero_value = tl.full([1], 0, tl.int64)
    threshold_128 = tl.full([1], 128, tl.int64)
    condition_128 = x_mod_480 < threshold_128
    value0 = tl.load(input_ptr0 + (128 * x_div_480 + x_mod_480), condition_128 & mask, eviction_policy='evict_last', other=0.0)
    value1 = tl.load(input_ptr1 + x_mod_480, condition_128 & mask, eviction_policy='evict_last', other=0.0)
    sum_128 = value0 + value1
    zero_like_sum_128 = tl.full(sum_128.shape, 0.0, sum_128.dtype)
    result_128 = tl.where(condition_128, sum_128, zero_like_sum_128)

    threshold_320 = tl.full([1], 320, tl.int64)
    condition_320 = (x_mod_480 >= threshold_128) & (x_mod_480 < threshold_320)
    value2 = tl.load(input_ptr2 + (192 * x_div_480 + (x_mod_480 - 128)), condition_320 & mask, eviction_policy='evict_last', other=0.0)
    value3 = tl.load(input_ptr3 + (x_mod_480 - 128), condition_320 & mask, eviction_policy='evict_last', other=0.0)
    sum_320 = value2 + value3
    zero_like_sum_320 = tl.full(sum_320.shape, 0.0, sum_320.dtype)
    result_320 = tl.where(condition_320, sum_320, zero_like_sum_320)

    threshold_416 = tl.full([1], 416, tl.int64)
    condition_416 = (x_mod_480 >= threshold_320) & (x_mod_480 < threshold_416)
    value4 = tl.load(input_ptr4 + (96 * x_div_480 + (x_mod_480 - 320)), condition_416 & mask, eviction_policy='evict_last', other=0.0)
    value5 = tl.load(input_ptr5 + (x_mod_480 - 320), condition_416 & mask, eviction_policy='evict_last', other=0.0)
    sum_416 = value4 + value5
    zero_like_sum_416 = tl.full(sum_416.shape, 0.0, sum_416.dtype)
    result_416 = tl.where(condition_416, sum_416, zero_like_sum_416)

    threshold_480 = tl.full([1], 480, tl.int64)
    condition_480 = x_mod_480 >= threshold_416
    value6 = tl.load(input_ptr6 + (64 * x_div_480 + (x_mod_480 - 416)), condition_480 & mask, eviction_policy='evict_last', other=0.0)
    value7 = tl.load(input_ptr7 + (x_mod_480 - 416), condition_480 & mask, eviction_policy='evict_last', other=0.0)
    sum_480 = value6 + value7
    zero_like_sum_480 = tl.full(sum_480.shape, 0.0, sum_480.dtype)
    result_480 = tl.where(condition_480, sum_480, zero_like_sum_480)

    final_result = tl.where(condition_416, result_416, result_480)
    final_result = tl.where(condition_320, result_320, final_result)
    final_result = tl.where(condition_128, result_128, final_result)

    tl.store(output_ptr0 + x_index, final_result, mask)