# From: 84_Gemm_BatchNorm_Scaling_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_0(
    input_ptr_mean, input_ptr_var, input_ptr_scale, 
    output_ptr_normalized, output_ptr_running_mean, 
    output_ptr_running_var, output_ptr_scaled, 
    kernel_size, input_num_elements, running_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 512
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    running_base = tl.arange(0, RBLOCK)[None, :]
    input_index_0 = input_index
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    
    for running_offset in range(0, running_num_elements, RBLOCK):
        running_index = running_offset + running_base
        running_mask = running_index < running_num_elements
        running_index_1 = running_index
        temp_input = tl.load(input_ptr_mean + (input_index_0 + 512 * running_index_1), 
                             running_mask & input_mask, 
                             eviction_policy='evict_first', 
                             other=0.0)
        temp_broadcast = tl.broadcast_to(temp_input, [XBLOCK, RBLOCK])
        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_reduce(
            temp_broadcast, temp_mean, temp_m2, temp_weight, running_offset == 0
        )
        temp_mean = tl.where(running_mask & input_mask, temp_mean_next, temp_mean)
        temp_m2 = tl.where(running_mask & input_mask, temp_m2_next, temp_m2)
        temp_weight = tl.where(running_mask & input_mask, temp_weight_next, temp_weight)
    
    temp_mean_final, temp_var_final, temp_weight_final = triton_helpers.welford(
        temp_mean, temp_m2, temp_weight, 1
    )
    temp_mean_final = temp_mean_final[:, None]
    temp_var_final = temp_var_final[:, None]
    temp_weight_final = temp_weight_final[:, None]
    
    tl.store(output_ptr_normalized + (input_index_0), temp_mean_final, input_mask)
    
    temp_scale = tl.load(input_ptr_scale + (input_index_0), input_mask, eviction_policy='evict_last')
    temp_bias = tl.load(input_ptr_var + (input_index_0), input_mask, eviction_policy='evict_last')
    
    kernel_size_float = kernel_size.to(tl.float32)
    mean_div_kernel_size = temp_mean_final / kernel_size_float
    epsilon = 1e-05
    mean_div_kernel_size_eps = mean_div_kernel_size + epsilon
    inv_sqrt = tl.extra.cuda.libdevice.rsqrt(mean_div_kernel_size_eps)
    
    momentum = (((512 * kernel_size) / 512) / ((tl.full([], -1.0, tl.float64)) + ((512 * kernel_size) / 512))).to(tl.float32)
    mean_scaled = mean_div_kernel_size * momentum
    scale_factor = 0.1
    mean_scaled_factor = mean_scaled * scale_factor
    
    momentum_running_mean = 0.9
    running_mean_scaled = temp_scale * momentum_running_mean
    running_mean_updated = mean_scaled_factor + running_mean_scaled
    
    temp_mean_scaled = temp_mean_final * scale_factor
    momentum_running_var = 0.9
    running_var_scaled = temp_bias * momentum_running_var
    running_var_updated = temp_mean_scaled + running_var_scaled
    
    tl.store(output_ptr_running_mean + (input_index_0), inv_sqrt, input_mask)
    tl.store(output_ptr_running_var + (input_index_0), running_mean_updated, input_mask)
    tl.store(output_ptr_scaled + (input_index_0), running_var_updated, input_mask)