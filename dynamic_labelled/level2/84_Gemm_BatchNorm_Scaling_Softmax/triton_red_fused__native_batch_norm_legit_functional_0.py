# From: 84_Gemm_BatchNorm_Scaling_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_0(
    input_ptr_mean, input_ptr_var, input_ptr_scale, 
    output_ptr_normalized, output_ptr_running_mean, output_ptr_running_var, 
    kernel_size, num_elements_x, num_elements_r, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements_x = 512
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < num_elements_x
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = x_index
    batch_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    batch_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    batch_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    
    for r_offset in range(0, num_elements_r, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < num_elements_r
        r1 = r_index
        input_data = tl.load(input_ptr_mean + (x0 + 512 * r1), rmask & x_mask, eviction_policy='evict_first', other=0.0)
        broadcast_input = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        batch_mean_next, batch_m2_next, batch_weight_next = triton_helpers.welford_reduce(
            broadcast_input, batch_mean, batch_m2, batch_weight, r_offset == 0
        )
        batch_mean = tl.where(rmask & x_mask, batch_mean_next, batch_mean)
        batch_m2 = tl.where(rmask & x_mask, batch_m2_next, batch_m2)
        batch_weight = tl.where(rmask & x_mask, batch_weight_next, batch_weight)
    
    mean, variance, weight = triton_helpers.welford(batch_mean, batch_m2, batch_weight, 1)
    mean = mean[:, None]
    variance = variance[:, None]
    weight = weight[:, None]
    
    tl.store(output_ptr_normalized + (x0), mean, x_mask)
    
    input_scale = tl.load(input_ptr_scale + (x0), x_mask, eviction_policy='evict_last')
    input_shift = tl.load(input_ptr_var + (x0), x_mask, eviction_policy='evict_last')
    
    kernel_size_float = kernel_size.to(tl.float32)
    normalized_mean = variance / kernel_size_float
    epsilon = 1e-05
    variance_with_epsilon = normalized_mean + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    
    scale_factor = (((512 * kernel_size) / 512) / ((tl.full([], -1.00000000000000, tl.float64)) + ((512 * kernel_size) / 512)))
    scale_factor = scale_factor.to(tl.float32)
    normalized_variance = normalized_mean * scale_factor
    
    momentum = 0.1
    adjusted_variance = normalized_variance * momentum
    
    running_mean_momentum = 0.9
    updated_running_mean = input_scale * running_mean_momentum
    updated_running_mean += adjusted_variance
    
    running_var_momentum = 0.9
    updated_running_var = mean * momentum
    updated_running_var += input_shift * running_var_momentum
    
    tl.store(output_ptr_running_mean + (x0), inv_std, x_mask)
    tl.store(output_ptr_running_mean + (x0), updated_running_mean, x_mask)
    tl.store(output_ptr_running_var + (x0), updated_running_var, x_mask)