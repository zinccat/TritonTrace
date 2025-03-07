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
    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    
    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_1 = reduction_index
        input_data = tl.load(input_ptr_mean + (input_index_0 + 512 * reduction_index_1), 
                             reduction_mask & input_mask, 
                             eviction_policy='evict_first', 
                             other=0.0)
        broadcasted_input = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_reduce(
            broadcasted_input, running_mean, running_m2, running_weight, reduction_offset == 0
        )
        running_mean = tl.where(reduction_mask & input_mask, running_mean_next, running_mean)
        running_m2 = tl.where(reduction_mask & input_mask, running_m2_next, running_m2)
        running_weight = tl.where(reduction_mask & input_mask, running_weight_next, running_weight)
    
    mean, variance, weight = triton_helpers.welford(running_mean, running_m2, running_weight, 1)
    mean = mean[:, None]
    variance = variance[:, None]
    weight = weight[:, None]
    
    tl.store(output_ptr_normalized + (input_index_0), mean, input_mask)
    
    input_scale = tl.load(input_ptr_var + (input_index_0), input_mask, eviction_policy='evict_last')
    input_offset_data = tl.load(input_ptr_bias + (input_index_0), input_mask, eviction_policy='evict_last')
    
    kernel_size_float = kernel_size.to(tl.float32)
    normalized_variance = variance / kernel_size_float
    epsilon = 1e-05
    variance_with_epsilon = normalized_variance + epsilon
    rsqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    
    scale_factor = (((512 * kernel_size) / 512) / ((tl.full([], -1.00000000000000, tl.float64)) + ((512 * kernel_size) / 512)))
    scale_factor_float = scale_factor.to(tl.float32)
    normalized_variance_scaled = normalized_variance * scale_factor_float
    swish_beta = 0.1
    swish_scale = normalized_variance_scaled * swish_beta
    
    momentum = 0.9
    scaled_input_scale = input_scale * momentum
    swish_offset = swish_scale + scaled_input_scale
    
    swish_weight = mean * swish_beta
    scaled_input_offset = input_offset_data * momentum
    swish_bias = swish_weight + scaled_input_offset
    
    tl.store(output_ptr_scale + (input_index_0), rsqrt_variance, input_mask)
    tl.store(output_ptr_normalized + (input_index_0), swish_offset, input_mask)
    tl.store(output_ptr_offset + (input_index_0), swish_bias, input_mask)