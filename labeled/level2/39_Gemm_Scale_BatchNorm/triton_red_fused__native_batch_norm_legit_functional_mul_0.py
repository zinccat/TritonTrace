# From: 39_Gemm_Scale_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_mul_0(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, 
    output_ptr_mean, output_ptr_var, output_ptr_scale, output_ptr_shift, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 512
    rnumel = 128
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = x_index
    input_var = tl.load(input_ptr_var + (x0), x_mask, eviction_policy='evict_last')
    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r1 = r_index
        input_data = tl.load(input_ptr_mean + (x0 + (512 * r1)), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        weighted_input = input_data * input_var
        broadcast_weighted_input = tl.broadcast_to(weighted_input, [XBLOCK, RBLOCK])
        
        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_reduce(
            broadcast_weighted_input, running_mean, running_m2, running_weight, r_offset == 0
        )
        
        running_mean = tl.where(r_mask & x_mask, running_mean_next, running_mean)
        running_m2 = tl.where(r_mask & x_mask, running_m2_next, running_m2)
        running_weight = tl.where(r_mask & x_mask, running_weight_next, running_weight)
    
    mean, variance, weight = triton_helpers.welford(running_mean, running_m2, running_weight, 1)
    mean = mean[:, None]
    variance = variance[:, None]
    
    tl.store(output_ptr_mean + (x0), mean, x_mask)
    tl.store(output_ptr_var + (x0), variance, x_mask)
    
    input_scale = tl.load(input_ptr_scale + (x0), x_mask, eviction_policy='evict_last')
    input_shift = tl.load(input_ptr_shift + (x0), x_mask, eviction_policy='evict_last')
    
    epsilon = 128.0
    epsilon_add = 1e-05
    rsqrt_input = variance / epsilon + epsilon_add
    rsqrt_result = tl.extra.cuda.libdevice.rsqrt(rsqrt_input)
    
    scale_factor = 1.0078740157480315
    scaled_variance = variance * scale_factor
    momentum = 0.1
    updated_scale = scaled_variance * momentum
    
    momentum_mean = 0.9
    updated_mean = input_scale * momentum_mean
    mean_update = updated_scale + updated_mean
    
    shift_momentum = 0.9
    updated_shift = input_shift * shift_momentum
    shift_update = updated_scale + updated_shift
    
    tl.store(output_ptr_scale + (x0), rsqrt_result, x_mask)
    tl.store(output_ptr_shift + (x0), mean_update, x_mask)
    tl.store(output_ptr_scale + (x0), shift_update, x_mask)