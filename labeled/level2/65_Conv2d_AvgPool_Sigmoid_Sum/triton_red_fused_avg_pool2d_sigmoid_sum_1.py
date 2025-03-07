# From: 65_Conv2d_AvgPool_Sigmoid_Sum

import triton
import triton.language as tl


@triton.jit
def triton_red_fused_avg_pool2d_sigmoid_sum_1(input_ptr, output_ptr1, output_ptr2, num_elements_x, num_elements_r, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    num_elements_x = 128
    num_elements_r = 3600
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_0 = x_indices
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, num_elements_r, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < num_elements_r
        r1 = r_indices % 15
        r2 = (r_indices // 15)
        r3 = r_indices
        temp0 = tl.load(input_ptr + ((2 * r1) + (60 * r2) + (14400 * x_indices_0)), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        temp1 = tl.load(input_ptr + (1 + (2 * r1) + (60 * r2) + (14400 * x_indices_0)), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        temp3 = tl.load(input_ptr + (30 + (2 * r1) + (60 * r2) + (14400 * x_indices_0)), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        temp5 = tl.load(input_ptr + (31 + (2 * r1) + (60 * r2) + (14400 * x_indices_0)), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        
        temp2 = temp1 + temp0
        temp4 = temp3 + temp2
        temp6 = temp5 + temp4
        
        avg_pool = 0.25
        temp8 = temp6 * avg_pool
        sigmoid_output = tl.sigmoid(temp8)
        broadcast_sigmoid = tl.broadcast_to(sigmoid_output, [XBLOCK, RBLOCK])
        
        temp12 = temp_sum + broadcast_sigmoid
        temp_sum = tl.where(r_mask & x_mask, temp12, temp_sum)
        
        tl.store(output_ptr1 + (r3 + (3616 * x_indices_0)), temp8, r_mask & x_mask)
    
    summed_result = tl.sum(temp_sum, 1)[:, None]
    tl.store(output_ptr2 + (x_indices_0), summed_result, x_mask)