# From: 27_Conv3d_HardSwish_ReLU_Softmax_Mean

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_hardswish_relu_1(in_ptr0, out_ptr0, out_ptr1, kernel_size0, kernel_size1, kernel_size2, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 16
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    reduction_index_2 = reduction_index
    input_index_3 = (input_index % kernel_size0)
    input_index_4 = input_index // kernel_size0
    input_index_5 = input_index
    loaded_value = tl.load(
        in_ptr0 + (
            input_index_3 + 
            ((-128) * input_index_4) + 
            ((-8) * reduction_index_2) + 
            ((-32) * input_index_4 * kernel_size2 * kernel_size2) + 
            ((-2) * reduction_index_2 * kernel_size2 * kernel_size2) + 
            4 * kernel_size1 * reduction_index_2 + 
            8 * kernel_size2 * reduction_index_2 + 
            64 * kernel_size1 * input_index_4 + 
            128 * kernel_size2 * input_index_4 + 
            kernel_size1 * reduction_index_2 * kernel_size2 * kernel_size2 + 
            ((-64) * kernel_size1 * kernel_size2 * input_index_4) + 
            ((-4) * kernel_size1 * kernel_size2 * reduction_index_2) + 
            16 * kernel_size1 * input_index_4 * kernel_size2 * kernel_size2
        ), 
        input_mask, 
        eviction_policy='evict_last', 
        other=0.0
    )
    bias = 3.0
    biased_value = loaded_value + bias
    zero = 0.0
    max_value = triton_helpers.maximum(biased_value, zero)
    upper_bound = 6.0
    clipped_value = triton_helpers.minimum(max_value, upper_bound)
    hardswish_value = loaded_value * clipped_value
    scale_factor = 0.16666666666666666
    scaled_value = hardswish_value * scale_factor
    zero_int32 = tl.full([1, 1], 0, tl.int32)
    max_scaled_value = triton_helpers.maximum(zero_int32, scaled_value)
    broadcast_max_scaled_value = tl.broadcast_to(max_scaled_value, [XBLOCK, RBLOCK])
    masked_broadcast_value = tl.where(input_mask, broadcast_max_scaled_value, float("-inf"))
    max_across_blocks = triton_helpers.max2(masked_broadcast_value, 1)[:, None]
    shifted_value = max_scaled_value - max_across_blocks
    exp_value = tl.math.exp(shifted_value)
    broadcast_exp_value = tl.broadcast_to(exp_value, [XBLOCK, RBLOCK])
    masked_exp_value = tl.where(input_mask, broadcast_exp_value, 0)
    sum_exp_values = tl.sum(masked_exp_value, 1)[:, None]
    tl.store(out_ptr0 + (input_index_5), max_across_blocks, input_mask)
    tl.store(out_ptr1 + (input_index_5), sum_exp_values, input_mask)