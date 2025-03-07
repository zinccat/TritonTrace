# From: 24_Conv3d_Min_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_min_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, kernel_size0, kernel_size1, kernel_size2, kernel_size3, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_mod_k0 = (input_index % kernel_size0)
    input_div_k0 = input_index // kernel_size0
    input_mod_16 = ((input_index // kernel_size3) % 16)
    
    temp1 = tl.load(in_ptr1 + (input_mod_16), input_mask, eviction_policy='evict_last')
    temp4_inf = tl.full([XBLOCK, RBLOCK], float("inf"), tl.float32)
    input_index_copy = input_index
    temp6_inf = tl.full([XBLOCK, RBLOCK], float("inf"), tl.float32)
    temp6_index_inf = tl.full([XBLOCK, RBLOCK], 9223372036854775807, tl.int64)
    
    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_copy = reduction_index
        
        temp0 = tl.load(
            in_ptr0 + (
                input_mod_k0 + 
                ((-8) * input_div_k0) + 
                4 * reduction_index_copy + 
                reduction_index_copy * kernel_size2 * kernel_size2 + 
                ((-4) * kernel_size2 * reduction_index_copy) + 
                ((-2) * input_div_k0 * kernel_size2 * kernel_size2) + 
                4 * kernel_size1 * input_div_k0 + 
                8 * kernel_size2 * input_div_k0 + 
                kernel_size1 * input_div_k0 * kernel_size2 * kernel_size2 + 
                ((-4) * kernel_size1 * kernel_size2 * input_div_k0)
            ), 
            reduction_mask & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        
        temp2 = temp0 + temp1
        temp3 = tl.broadcast_to(temp2, [XBLOCK, RBLOCK])
        temp5 = triton_helpers.minimum(temp4_inf, temp3)
        temp4_inf = tl.where(reduction_mask & input_mask, temp5, temp4_inf)
        
        temp6_next, temp6_index_next = triton_helpers.minimum_with_index(
            temp6_inf, temp6_index_inf, temp3, reduction_index
        )
        temp6_inf = tl.where(reduction_mask & input_mask, temp6_next, temp6_inf)
        temp6_index_inf = tl.where(reduction_mask & input_mask, temp6_index_next, temp6_index_inf)
    
    min_temp4 = triton_helpers.min2(temp4_inf, 1)[:, None]
    min_temp6_val, min_temp6_idx = triton_helpers.min_with_index(temp6_inf, temp6_index_inf, 1)
    min_temp6 = min_temp6_idx[:, None]
    
    tl.store(out_ptr0 + (input_index_copy), min_temp4, input_mask)
    tl.store(out_ptr1 + (input_index_copy), min_temp6, input_mask)