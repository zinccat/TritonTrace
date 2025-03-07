# From: 94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_backward_0(
    input_grad_ptr, input_ptr, running_mean_ptr, running_var_ptr, output_grad_ptr, output_ptr, 
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    
    row_index = rindex
    col_index = xindex
    col_mod_index = xindex % 32
    
    grad_input = tl.load(input_grad_ptr + (row_index + 32 * col_index), xmask, other=0.0)
    input_data = tl.load(input_ptr + (row_index + 32 * col_index), xmask, other=0.0)
    running_mean = tl.load(running_mean_ptr + (row_index + 32 * col_mod_index), xmask, eviction_policy='evict_last', other=0.0)
    running_var = tl.load(running_var_ptr + (row_index + 32 * col_mod_index), xmask, eviction_policy='evict_last', other=0.0)
    
    normalized_input = input_data + running_mean
    lower_bound = -1.0
    upper_bound = 1.0
    clamped_input = triton_helpers.maximum(normalized_input, lower_bound)
    clamped_input = triton_helpers.minimum(clamped_input, upper_bound)
    
    exp_input = tl.math.exp(clamped_input)
    log1p_exp_input = tl.extra.cuda.libdevice.log1p(exp_input)
    hard_tanh_output = tl.where(clamped_input > 20.0, clamped_input, log1p_exp_input)
    tanh_output = tl.extra.cuda.libdevice.tanh(hard_tanh_output)
    mish_output = clamped_input * tanh_output
    
    grad_input_scaled = grad_input * mish_output
    grad_input_scaled_var = grad_input_scaled * running_var
    
    broadcast_grad_input_scaled_var = tl.broadcast_to(grad_input_scaled_var, [XBLOCK, RBLOCK])
    masked_grad_input_scaled_var = tl.where(xmask, broadcast_grad_input_scaled_var, 0)
    sum_grad_input_scaled_var = tl.sum(masked_grad_input_scaled_var, 1)[:, None]
    
    grad_input_scaled_broadcast = tl.broadcast_to(grad_input_scaled, [XBLOCK, RBLOCK])
    masked_grad_input_scaled = tl.where(xmask, grad_input_scaled_broadcast, 0)
    sum_grad_input_scaled = tl.sum(masked_grad_input_scaled, 1)[:, None]
    
    tl.store(output_grad_ptr + (col_index), sum_grad_input_scaled_var, xmask)
    tl.store(output_ptr + (col_index), sum_grad_input_scaled, xmask)