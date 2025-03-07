# From: 97_Matmul_BatchNorm_BiasAdd_Divide_Swish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_0(
    input_ptr_mean, input_ptr_var, input_ptr_bias, 
    output_ptr_normalized, output_ptr_scale, output_ptr_offset, 
    kernel_size, input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 512
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_0 = input_index
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    
    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_1 = reduction_index
        temp_input = tl.load(input_ptr_mean + (input_index_0 + 512 * reduction_index_1), 
                             reduction_mask & input_mask, 
                             eviction_policy='evict_first', 
                             other=0.0)
        temp_broadcast = tl.broadcast_to(temp_input, [XBLOCK, RBLOCK])
        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_reduce(
            temp_broadcast, temp_mean, temp_m2, temp_weight, reduction_offset == 0
        )
        temp_mean = tl.where(reduction_mask & input_mask, temp_mean_next, temp_mean)
        temp_m2 = tl.where(reduction_mask & input_mask, temp_m2_next, temp_m2)
        temp_weight = tl.where(reduction_mask & input_mask, temp_weight_next, temp_weight)
    
    temp_mean_final, temp_var_final, temp_weight_final = triton_helpers.welford(
        temp_mean, temp_m2, temp_weight, 1
    )
    temp_mean_final = temp_mean_final[:, None]
    temp_var_final = temp_var_final[:, None]
    temp_weight_final = temp_weight_final[:, None]
    
    tl.store(output_ptr_normalized + (input_index_0), temp_mean_final, input_mask)
    
    temp_scale = tl.load(input_ptr_var + (input_index_0), input_mask, eviction_policy='evict_last')
    temp_bias = tl.load(input_ptr_bias + (input_index_0), input_mask, eviction_policy='evict_last')
    
    kernel_size_float = kernel_size.to(tl.float32)
    temp_var_div = temp_var_final / kernel_size_float
    epsilon = 1e-05
    temp_var_add_epsilon = temp_var_div + epsilon
    temp_var_rsqrt = tl.extra.cuda.libdevice.rsqrt(temp_var_add_epsilon)
    
    temp_var_scale_factor = (((512 * kernel_size) / 512) / 
                             ((tl.full([], -1.00000000000000, tl.float64)) + 
                              ((512 * kernel_size) / 512))).to(tl.float32)
    temp_var_scaled = temp_var_div * temp_var_scale_factor
    swish_beta = 0.1
    temp_var_swish = temp_var_scaled * swish_beta
    
    momentum = 0.9
    temp_scale_momentum = temp_scale * momentum
    temp_scale_swish = temp_var_swish + temp_scale_momentum
    
    temp_bias_momentum = temp_bias * momentum
    temp_bias_swish = temp_mean_final * swish_beta + temp_bias_momentum
    
    tl.store(output_ptr_scale + (input_index_0), temp_var_rsqrt, input_mask)
    tl.store(output_ptr_normalized + (input_index_0), temp_scale_swish, input_mask)
    tl.store(output_ptr_offset + (input_index_0), temp_bias_swish, input_mask)