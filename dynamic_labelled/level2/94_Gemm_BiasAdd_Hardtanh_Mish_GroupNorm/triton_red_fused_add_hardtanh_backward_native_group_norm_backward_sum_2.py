# From: 94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_hardtanh_backward_native_group_norm_backward_sum_2(
    input_grad_ptr, input_data_ptr, bias_ptr, gamma_ptr, beta_ptr, mean_ptr, 
    output_grad_ptr0, output_grad_ptr1, output_grad_ptr2, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    bias = tl.load(bias_ptr + (x3), xmask, eviction_policy='evict_last')
    x1 = xindex // 32
    grad_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    grad_accumulator2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    grad_accumulator3 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        input_grad = tl.load(input_grad_ptr + (x3 + 1024 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        input_data = tl.load(input_data_ptr + (x3 + 1024 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        gamma = tl.load(gamma_ptr + (x1 + 32 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        beta = tl.load(beta_ptr + (x1 + 32 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        mean = tl.load(mean_ptr + (x3 + 1024 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        
        combined_input = input_data + bias
        lower_bound = -1.0
        upper_bound = 1.0
        clipped_input = triton_helpers.maximum(combined_input, lower_bound)
        clipped_input = triton_helpers.minimum(clipped_input, upper_bound)
        
        threshold = 20.0
        exp_clipped_input = tl.math.exp(clipped_input)
        log1p_exp = tl.extra.cuda.libdevice.log1p(exp_clipped_input)
        log_clipped_input = tl.where(clipped_input > threshold, clipped_input, log1p_exp)
        
        tanh_log = tl.extra.cuda.libdevice.tanh(log_clipped_input)
        derivative = clipped_input * tanh_log
        grad_input = input_grad * derivative
        grad_gamma = grad_input * gamma
        grad_beta = grad_input - grad_gamma
        
        grad_accumulator += tl.broadcast_to(grad_beta, [XBLOCK, RBLOCK])
        grad_accumulator2 += tl.where(combined_input <= lower_bound | combined_input >= upper_bound, 0.0, mean)
        grad_accumulator3 += tl.broadcast_to(input_grad, [XBLOCK, RBLOCK])
    
    grad_accumulator = tl.sum(grad_accumulator, 1)[:, None]
    grad_accumulator2 = tl.sum(grad_accumulator2, 1)[:, None]
    grad_accumulator3 = tl.sum(grad_accumulator3, 1)[:, None]
    
    tl.store(output_grad_ptr0 + (x3), grad_accumulator, xmask)
    tl.store(output_grad_ptr1 + (x3), grad_accumulator2, xmask)
    tl.store(output_grad_ptr2 + (x3), grad_accumulator3, xmask)