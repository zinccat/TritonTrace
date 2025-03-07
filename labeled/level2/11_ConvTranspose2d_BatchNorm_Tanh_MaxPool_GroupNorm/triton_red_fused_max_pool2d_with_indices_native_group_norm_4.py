# From: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused_max_pool2d_with_indices_native_group_norm_4(
    input_ptr, output_ptr_max, output_ptr_indices, output_ptr_mean, output_ptr_var, 
    num_elements_input, num_elements_output, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements_input = 512
    num_elements_output = 16384
    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < num_elements_input
    output_base_indices = tl.arange(0, RBLOCK)[None, :]
    
    input_index = input_indices
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    m2_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    
    for output_offset in range(0, num_elements_output, RBLOCK):
        output_indices = output_offset + output_base_indices
        output_mask = output_indices < num_elements_output
        
        output_index_mod = output_indices % 32
        output_index_div = (output_indices // 32)
        output_index = output_indices
        
        input_value_0 = tl.load(
            input_ptr + ((2 * output_index_mod) + (128 * output_index_div) + (65536 * input_index)), 
            output_mask & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        input_value_1 = tl.load(
            input_ptr + (1 + (2 * output_index_mod) + (128 * output_index_div) + (65536 * input_index)), 
            output_mask & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        input_value_2 = tl.load(
            input_ptr + (64 + (2 * output_index_mod) + (128 * output_index_div) + (65536 * input_index)), 
            output_mask & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        input_value_3 = tl.load(
            input_ptr + (65 + (2 * output_index_mod) + (128 * output_index_div) + (65536 * input_index)), 
            output_mask & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        
        max_value_1_2 = triton_helpers.maximum(input_value_1, input_value_0)
        max_value_2_3 = triton_helpers.maximum(input_value_2, max_value_1_2)
        max_value_3_4 = triton_helpers.maximum(input_value_3, max_value_2_3)
        
        index_1_greater_0 = input_value_1 > input_value_0
        index_1_mask = tl.full([1, 1], 1, tl.int8)
        index_0_mask = tl.full([1, 1], 0, tl.int8)
        index_mask_1_2 = tl.where(index_1_greater_0, index_1_mask, index_0_mask)
        
        index_2_greater_1 = input_value_2 > max_value_1_2
        index_2_mask = tl.full([1, 1], 2, tl.int8)
        index_mask_2_3 = tl.where(index_2_greater_1, index_2_mask, index_mask_1_2)
        
        index_3_greater_2 = input_value_3 > max_value_2_3
        index_3_mask = tl.full([1, 1], 3, tl.int8)
        index_mask_3_4 = tl.where(index_3_greater_2, index_3_mask, index_mask_2_3)
        
        broadcast_max_value = tl.broadcast_to(max_value_3_4, [XBLOCK, RBLOCK])
        
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            broadcast_max_value, mean_accumulator, m2_accumulator, weight_accumulator, output_offset == 0
        )
        
        mean_accumulator = tl.where(output_mask & input_mask, mean_next, mean_accumulator)
        m2_accumulator = tl.where(output_mask & input_mask, m2_next, m2_accumulator)
        weight_accumulator = tl.where(output_mask & input_mask, weight_next, weight_accumulator)
        
        tl.store(output_ptr_max + (output_index + (16384 * input_index)), max_value_3_4, output_mask & input_mask)
        tl.store(output_ptr_indices + (output_index + (16384 * input_index)), index_mask_3_4, output_mask & input_mask)
    
    mean_final, m2_final, weight_final = triton_helpers.welford(
        mean_accumulator, m2_accumulator, weight_accumulator, 1
    )
    
    mean_final_broadcast = mean_final[:, None]
    m2_final_broadcast = m2_final[:, None]
    
    tl.store(output_ptr_mean + (input_index), mean_final_broadcast, input_mask)
    tl.store(output_ptr_var + (input_index), m2_final_broadcast, input_mask)
    
    normalization_factor = 16384.0
    variance = m2_final_broadcast / normalization_factor
    epsilon = 1e-05
    variance_with_epsilon = variance + epsilon
    inverse_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    
    tl.store(output_ptr_var + (input_index), inverse_sqrt_variance, input_mask)