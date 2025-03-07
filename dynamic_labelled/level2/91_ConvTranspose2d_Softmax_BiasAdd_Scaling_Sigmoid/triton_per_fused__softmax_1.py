# From: 91_ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_1per_fused__softmax_1(in_ptr0, out_ptr0, out_ptr1, kernel_size0, kernel_size1, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 64
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    reduction_index_2 = reduction_index
    kernel_index_0 = (input_index % kernel_size0)
    kernel_index_1 = input_index // kernel_size0
    flat_input_index = input_index
    loaded_values = tl.load(
        in_ptr0 + (reduction_index_2 + kernel_index_0 + 64 * kernel_index_1 + 4 * kernel_size1 * reduction_index_2 + 4 * reduction_index_2 * kernel_size1 * kernel_size1 + 256 * kernel_size1 * kernel_index_1 + 256 * kernel_index_1 * kernel_size1 * kernel_size1),
        input_mask,
        eviction_policy='evict_last',
        other=0.0
    )
    broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
    masked_values = tl.where(input_mask, broadcasted_values, float("-inf"))
    max_values = triton_helpers.max2(masked_values, 1)[:, None]
    shifted_values = loaded_values - max_values
    exp_values = tl.math.exp(shifted_values)
    broadcasted_exp_values = tl.broadcast_to(exp_values, [XBLOCK, RBLOCK])
    masked_exp_values = tl.where(input_mask, broadcasted_exp_values, 0)
    sum_exp_values = tl.sum(masked_exp_values, 1)[:, None]
    tl.store(out_ptr0 + (flat_input_index), max_values, input_mask)
    tl.store(out_ptr1 + (flat_input_index), sum_exp_values, input_mask)