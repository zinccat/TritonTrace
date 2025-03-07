# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_54(
    input_ptr_mean, input_ptr_var, input_ptr_gamma, 
    output_ptr_mean, output_ptr_var, output_ptr_gamma, 
    output_ptr_beta, output_ptr_output, 
    total_elements, reduction_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    total_elements = 352
    reduction_elements = 7840
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    
    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_elements
        r_channel = (r_indices % 784)
        r_sample = r_indices // 784
        input_data = tl.load(
            input_ptr_mean + (r_channel + 784 * x_indices_flat + 275968 * r_sample), 
            r_mask & x_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        broadcast_input = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_reduce(
            broadcast_input, running_mean, running_m2, running_weight, r_offset == 0
        )
        running_mean = tl.where(r_mask & x_mask, running_mean_next, running_mean)
        running_m2 = tl.where(r_mask & x_mask, running_m2_next, running_m2)
        running_weight = tl.where(r_mask & x_mask, running_weight_next, running_weight)
    
    mean, variance, weight = triton_helpers.welford(running_mean, running_m2, running_weight, 1)
    mean_broadcast = mean[:, None]
    variance_broadcast = variance[:, None]
    weight_broadcast = weight[:, None]
    
    tl.store(output_ptr_mean + (x_indices_flat), mean_broadcast, x_mask)
    tl.store(output_ptr_var + (x_indices_flat), variance_broadcast, x_mask)
    
    gamma = tl.load(input_ptr_gamma + (x_indices_flat), x_mask, eviction_policy='evict_last')
    beta = tl.load(input_ptr_var + (x_indices_flat), x_mask, eviction_policy='evict_last')
    
    num_samples = 7840.0
    epsilon = 1e-05
    variance_adjusted = variance / num_samples + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_adjusted)
    
    momentum = 0.1
    running_mean_scaled = mean * momentum
    gamma_scaled = gamma * 0.9
    adjusted_mean = running_mean_scaled + gamma_scaled
    
    variance_scaled = variance * 1.0001275672917465
    variance_scaled_momentum = variance_scaled * momentum
    adjusted_variance = variance_scaled_momentum + gamma_scaled
    
    tl.store(output_ptr_gamma + (x_indices_flat), inv_stddev, x_mask)
    tl.store(output_ptr_beta + (x_indices_flat), adjusted_mean, x_mask)
    tl.store(output_ptr_output + (x_indices_flat), adjusted_variance, x_mask)