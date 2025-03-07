# From: 43_Conv3d_Max_LogSumExp_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_exp_logsumexp_sub_2poi_fused_exp_logsumexp_sub_2(in_out_ptr0, in_ptr0, in_ptr1, kernel_size0, kernel_size1, kernel_size2, kernel_size3, num_elements, XBLOCK : tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    linear_index = indices
    index_mod_k0 = indices % kernel_size0
    index_div_k1 = indices // kernel_size1
    
    output_value = tl.load(in_out_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    input_value0 = tl.load(in_ptr0 + (index_mod_k0 + index_div_k1 * (kernel_size3 // 2) * (kernel_size3 // 2) * (kernel_size2 // 2)), mask, eviction_policy='evict_last')
    input_value1 = tl.load(in_ptr1 + (index_mod_k0 + index_div_k1 * (kernel_size3 // 2) * (kernel_size3 // 2) * (kernel_size2 // 2)), mask, eviction_policy='evict_last')
    
    log_input_value0 = tl.math.log(input_value0)
    abs_input_value1 = tl.math.abs(input_value1)
    inf_value = float("inf")
    is_inf = abs_input_value1 == inf_value
    zero_value = 0.0
    adjusted_input_value1 = tl.where(is_inf, zero_value, input_value1)
    
    log_sum_exp = log_input_value0 + adjusted_input_value1
    difference = output_value - log_sum_exp
    exp_difference = tl.math.exp(difference)
    
    tl.store(in_out_ptr0 + (linear_index), exp_difference, mask)