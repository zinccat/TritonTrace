# From: 24_Conv3d_Min_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_min_0red_fused_convolution_min_0(
    input_ptr0, input_ptr1, output_ptr0, output_ptr1, kernel_size0, kernel_size1, kernel_size2, kernel_size3, 
    x_num_elements, r_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_mod_k0 = (x_index % kernel_size0)
    x_div_k0 = x_index // kernel_size0
    x_mod_16 = ((x_index // kernel_size3) % 16)
    
    temp_input1 = tl.load(input_ptr1 + (x_mod_16), x_mask, eviction_policy='evict_last')
    temp_min_inf = tl.full([XBLOCK, RBLOCK], float("inf"), tl.float32)
    x_index_flat = x_index
    temp_min_inf_final = tl.full([XBLOCK, RBLOCK], float("inf"), tl.float32)
    temp_min_index = tl.full([XBLOCK, RBLOCK], 9223372036854775807, tl.int64)
    
    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r_index_flat = rindex
        
        temp_input0 = tl.load(
            input_ptr0 + (
                x_mod_k0 + 
                ((-8) * x_div_k0) + 
                4 * r_index_flat + 
                r_index_flat * kernel_size2 * kernel_size2 + 
                ((-4) * kernel_size2 * r_index_flat) + 
                ((-2) * x_div_k0 * kernel_size2 * kernel_size2) + 
                4 * kernel_size1 * x_div_k0 + 
                8 * kernel_size2 * x_div_k0 + 
                kernel_size1 * x_div_k0 * kernel_size2 * kernel_size2 + 
                ((-4) * kernel_size1 * kernel_size2 * x_div_k0)
            ), 
            rmask & x_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        
        temp_sum = temp_input0 + temp_input1
        temp_broadcast = tl.broadcast_to(temp_sum, [XBLOCK, RBLOCK])
        temp_min = triton_helpers.minimum(temp_min_inf, temp_broadcast)
        temp_min_inf = tl.where(rmask & x_mask, temp_min, temp_min_inf)
        
        temp_min_next, temp_min_index_next = triton_helpers.minimum_with_index(
            temp_min_inf_final, temp_min_index, temp_broadcast, r_index
        )
        
        temp_min_inf_final = tl.where(rmask & x_mask, temp_min_next, temp_min_inf_final)
        temp_min_index = tl.where(rmask & x_mask, temp_min_index_next, temp_min_index)
    
    temp_min_result = triton_helpers.min2(temp_min_inf, 1)[:, None]
    temp_min_value, temp_min_index_result = triton_helpers.min_with_index(
        temp_min_inf_final, temp_min_index, 1
    )
    temp_min_index_result = temp_min_index_result[:, None]
    
    tl.store(output_ptr0 + (x_index_flat), temp_min_result, x_mask)
    tl.store(output_ptr1 + (x_index_flat), temp_min_index_result, x_mask)