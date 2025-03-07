# From: 35_Conv2d_Subtract_HardSwish_MaxPool_Mish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_add_fill_max_pool2d_with_indices_mish_mul_sigmoid_sub_1(
    input_ptr, output_ptr_indices, output_ptr_mish, output_ptr_hard_swish, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)

    col_index = block_indices % 15
    row_index = block_indices // 15
    channel_index = block_indices // 3600
    linear_index = block_indices % 3600
    global_index = block_indices

    input_val_0 = tl.load(input_ptr + ((2 * col_index) + (60 * row_index)), None, eviction_policy='evict_last')
    input_val_1 = tl.load(input_ptr + (1 + (2 * col_index) + (60 * row_index)), None, eviction_policy='evict_last')
    input_val_7 = tl.load(input_ptr + (30 + (2 * col_index) + (60 * row_index)), None, eviction_policy='evict_last')
    input_val_12 = tl.load(input_ptr + (31 + (2 * col_index) + (60 * row_index)), None, eviction_policy='evict_last')

    is_val1_greater = input_val_1 > input_val_0
    max_val_01 = tl.where(is_val1_greater, tl.full([1], 1, tl.int8), tl.full([1], 0, tl.int8))
    max_val_01 = triton_helpers.maximum(input_val_1, input_val_0)

    is_val7_greater = input_val_7 > max_val_01
    max_val_071 = tl.where(is_val7_greater, tl.full([1], 2, tl.int8), max_val_01)
    max_val_071 = triton_helpers.maximum(input_val_7, max_val_01)

    is_val12_greater = input_val_12 > max_val_071
    max_val_0712 = tl.where(is_val12_greater, tl.full([1], 3, tl.int8), max_val_071)
    max_val_0712 = triton_helpers.maximum(input_val_12, max_val_071)

    threshold = 20.0
    is_max_greater_than_threshold = max_val_0712 > threshold
    exp_max = tl.math.exp(max_val_0712)
    log1p_exp_max = tl.extra.cuda.libdevice.log1p(exp_max)
    log_max = tl.where(is_max_greater_than_threshold, max_val_0712, log1p_exp_max)

    tanh_log_max = tl.extra.cuda.libdevice.tanh(log_max)
    mish_output = max_val_0712 * tanh_log_max

    sigmoid_max = tl.sigmoid(max_val_0712)
    sigmoid_mul_max = max_val_0712 * sigmoid_max

    tanh_squared = tanh_log_max * tanh_log_max
    one_minus_tanh_squared = 1.0 - tanh_squared
    hard_swish_output = sigmoid_mul_max * one_minus_tanh_squared + tanh_log_max

    tl.store(output_ptr_indices + (linear_index + (3712 * channel_index)), max_val_0712, None)
    tl.store(output_ptr_mish + (global_index), mish_output, None)
    tl.store(output_ptr_hard_swish + (linear_index + (3616 * channel_index)), hard_swish_output, None)