# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_22(
    input_ptr_mean, input_ptr_var, input_ptr_input, 
    output_ptr_mean, output_ptr_var, output_ptr_input, 
    output_ptr_normalized, output_ptr_scale, 
    num_elements_input, num_elements_reduction, 
    BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_R: tl.constexpr
):
    num_elements_input = 192
    num_elements_reduction = 31360
    offset_x = tl.program_id(0) * BLOCK_SIZE_X
    index_x = offset_x + tl.arange(0, BLOCK_SIZE_X)[:, None]
    mask_x = index_x < num_elements_input
    base_r = tl.arange(0, BLOCK_SIZE_R)[None, :]
    index_x0 = index_x
    running_mean = tl.zeros([BLOCK_SIZE_X, BLOCK_SIZE_R], tl.float32)
    running_m2 = tl.zeros([BLOCK_SIZE_X, BLOCK_SIZE_R], tl.float32)
    running_weight = tl.zeros([BLOCK_SIZE_X, BLOCK_SIZE_R], tl.float32)
    
    for offset_r in range(0, num_elements_reduction, BLOCK_SIZE_R):
        index_r = offset_r + base_r
        mask_r = index_r < num_elements_reduction
        index_r1 = (index_r % 3136)
        index_r2 = index_r // 3136
        input_value = tl.load(
            input_ptr_mean + (index_r1 + 3136 * index_x0 + 602112 * index_r2), 
            mask_r & mask_x, 
            eviction_policy='evict_first', 
            other=0.0
        )
        broadcast_input = tl.broadcast_to(input_value, [BLOCK_SIZE_X, BLOCK_SIZE_R])
        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_reduce(
            broadcast_input, running_mean, running_m2, running_weight, offset_r == 0
        )
        running_mean = tl.where(mask_r & mask_x, running_mean_next, running_mean)
        running_m2 = tl.where(mask_r & mask_x, running_m2_next, running_m2)
        running_weight = tl.where(mask_r & mask_x, running_weight_next, running_weight)
    
    final_mean, final_var, final_weight = triton_helpers.welford(
        running_mean, running_m2, running_weight, 1
    )
    final_mean = final_mean[:, None]
    final_var = final_var[:, None]
    tl.store(output_ptr_mean + (index_x0), final_mean, mask_x)
    tl.store(output_ptr_var + (index_x0), final_var, mask_x)
    
    input_scale = tl.load(input_ptr_input + (index_x0), mask_x, eviction_policy='evict_last')
    input_shift = tl.load(input_ptr_var + (index_x0), mask_x, eviction_policy='evict_last')
    
    num_reduction_elements = 31360.0
    variance_epsilon = 1e-05
    normalized_variance = final_var / num_reduction_elements + variance_epsilon
    rsqrt_variance = tl.extra.cuda.libdevice.rsqrt(normalized_variance)
    
    momentum = 0.1
    running_mean_scaled = final_mean * momentum
    input_scale_momentum = input_scale * 0.9
    updated_mean = running_mean_scaled + input_scale_momentum
    
    scale_factor = 1.0000318887719635
    variance_scaled = final_var * scale_factor * momentum
    updated_variance = variance_scaled + input_scale_momentum
    
    tl.store(output_ptr_normalized + (index_x0), rsqrt_variance, mask_x)
    tl.store(output_ptr_scale + (index_x0), updated_mean, mask_x)
    tl.store(output_ptr_input + (index_x0), updated_variance, mask_x)