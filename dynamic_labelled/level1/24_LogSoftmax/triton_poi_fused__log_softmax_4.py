# From: 24_LogSoftmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__log_softmax_4(input_ptr_max, input_ptr_mean, input_ptr_sum_exp, output_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    element_index = index
    batch_index = index // kernel_size
    
    max_value = tl.load(input_ptr_max + (element_index), mask, eviction_policy='evict_last')
    mean_value = tl.load(input_ptr_mean + (batch_index), mask, eviction_policy='evict_last')
    sum_exp_value = tl.load(input_ptr_sum_exp + (batch_index), mask, eviction_policy='evict_last')
    
    subtracted_max = max_value - mean_value
    log_sum_exp = tl.math.log(sum_exp_value)
    log_softmax_value = subtracted_max - log_sum_exp
    
    tl.store(output_ptr + (element_index), log_softmax_value, mask)