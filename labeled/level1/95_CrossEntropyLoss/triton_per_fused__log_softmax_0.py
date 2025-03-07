# From: 95_CrossEntropyLoss

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused__log_softmax_0(input_ptr, output_ptr_max, output_ptr_sum, num_elements, num_classes, BLOCK_SIZE : tl.constexpr):
    num_classes = 10
    BLOCK_SIZE_CLASS: tl.constexpr = 16
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:, None]
    tl.full([BLOCK_SIZE, BLOCK_SIZE_CLASS], True, tl.int1)
    class_indices = tl.arange(0, BLOCK_SIZE_CLASS)[None, :]
    class_mask = class_indices < num_classes
    class_offset = class_indices
    input_offset = block_indices
    input_values = tl.load(input_ptr + (class_offset + (10 * input_offset)), class_mask, other=0.0)
    broadcasted_values = tl.broadcast_to(input_values, [BLOCK_SIZE, BLOCK_SIZE_CLASS])
    masked_values = tl.where(class_mask, broadcasted_values, float("-inf"))
    max_values = triton_helpers.max2(masked_values, 1)[:, None]
    shifted_values = input_values - max_values
    exp_values = tl.math.exp(shifted_values)
    broadcasted_exp_values = tl.broadcast_to(exp_values, [BLOCK_SIZE, BLOCK_SIZE_CLASS])
    masked_exp_values = tl.where(class_mask, broadcasted_exp_values, 0)
    sum_exp_values = tl.sum(masked_exp_values, 1)[:, None]
    tl.store(output_ptr_max + (block_indices), max_values, None)
    tl.store(output_ptr_sum + (block_indices), sum_exp_values, None)