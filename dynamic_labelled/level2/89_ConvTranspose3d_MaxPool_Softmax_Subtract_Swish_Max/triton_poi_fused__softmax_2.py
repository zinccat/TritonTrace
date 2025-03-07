# From: 89_ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax_2poi_fused__softmax_2(
    in_out_ptr0, in_ptr0, in_ptr1, kernel_size_0, kernel_size_1, kernel_size_2, kernel_size_3, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    linear_index = indices
    index_0 = indices % kernel_size_0
    index_2 = indices // kernel_size_1

    loaded_value = tl.load(in_out_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    max_value = tl.load(in_ptr0 + (index_0 + kernel_size_2 * index_2 * kernel_size_3 * kernel_size_3), mask, eviction_policy='evict_last')
    sum_exp_values = tl.load(in_ptr1 + (index_0 + kernel_size_2 * index_2 * kernel_size_3 * kernel_size_3), mask, eviction_policy='evict_last')

    shifted_value = loaded_value - max_value
    exp_value = tl.math.exp(shifted_value)
    softmax_result = exp_value / sum_exp_values

    tl.store(in_out_ptr0 + (linear_index), softmax_result, mask)