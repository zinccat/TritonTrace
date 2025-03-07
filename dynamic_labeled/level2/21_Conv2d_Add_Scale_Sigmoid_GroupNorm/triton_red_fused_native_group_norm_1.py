# From: 21_Conv2d_Add_Scale_Sigmoid_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_1(input_ptr_mean, input_ptr_var, input_ptr_weight, output_ptr_mean, output_ptr_var, output_ptr_inv_std, kernel_size_0, kernel_size_1, num_elements_x, num_elements_r, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < num_elements_x
    r_base = tl.arange(0, RBLOCK)[None, :]
    x4 = x_index
    x0 = (x_index % 8)
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    
    for r_offset in range(0, num_elements_r, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < num_elements_r
        r2 = (r_index % kernel_size_0)
        r3 = r_index // kernel_size_0
        temp_input_mean = tl.load(input_ptr_mean + (((-2)*(triton_helpers.div_floor_integer(r2,  (-2) + kernel_size_1))) + 4*r3 + 8*x4 + kernel_size_1*(triton_helpers.div_floor_integer(r2,  (-2) + kernel_size_1)) + r3*kernel_size_1*kernel_size_1 + ((-8)*kernel_size_1*x4) + ((-4)*kernel_size_1*r3) + 2*x4*kernel_size_1*kernel_size_1 + ((r2 % ((-2) + kernel_size_1)))), rmask & x_mask, eviction_policy='evict_last', other=0.0)
        temp_input_var = tl.load(input_ptr_var + (r3 + 2*x0), rmask & x_mask, eviction_policy='evict_last', other=0.0)
        temp_input_weight = tl.load(input_ptr_weight + (r3 + 2*x0), rmask & x_mask, eviction_policy='evict_last', other=0.0)
        
        temp_sum = temp_input_mean + temp_input_var
        temp_product = temp_sum * temp_input_weight
        temp_sigmoid = tl.sigmoid(temp_product)
        temp_broadcast = tl.broadcast_to(temp_sigmoid, [XBLOCK, RBLOCK])
        
        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_reduce(
            temp_broadcast, temp_mean, temp_m2, temp_weight, r_offset == 0
        )
        
        temp_mean = tl.where(rmask & x_mask, temp_mean_next, temp_mean)
        temp_m2 = tl.where(rmask & x_mask, temp_m2_next, temp_m2)
        temp_weight = tl.where(rmask & x_mask, temp_weight_next, temp_weight)
    
    temp_mean_final, temp_m2_final, temp_weight_final = triton_helpers.welford(
        temp_mean, temp_m2, temp_weight, 1
    )
    
    temp_mean_final = temp_mean_final[:, None]
    temp_m2_final = temp_m2_final[:, None]
    temp_weight_final = temp_weight_final[:, None]
    
    tl.store(output_ptr_mean + (x4), temp_mean_final, x_mask)
    tl.store(output_ptr_var + (x4), temp_m2_final, x_mask)
    
    epsilon = 1e-05
    temp_var_adjusted = (8 + ((-8)*kernel_size_1) + 2*kernel_size_1*kernel_size_1) * ((8 + ((-8)*kernel_size_1) + 2*kernel_size_1*kernel_size_1) > (tl.full([], 0.0, tl.float64)))
    temp_var_adjusted = tl.full([], 0.0, tl.float64) * ((tl.full([], 0.0, tl.float64)) >= temp_var_adjusted) + temp_var_adjusted
    temp_var_adjusted = temp_var_adjusted.to(tl.float32)
    
    temp_std = temp_m2_final / temp_var_adjusted
    temp_inv_std = tl.extra.cuda.libdevice.rsqrt(temp_std + epsilon)
    
    tl.store(output_ptr_inv_std + (x4), temp_inv_std, x_mask)