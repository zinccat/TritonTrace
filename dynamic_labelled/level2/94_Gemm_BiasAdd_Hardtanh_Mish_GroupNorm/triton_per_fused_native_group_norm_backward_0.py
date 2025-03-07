# From: 94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_backward_0(
    input_grad_ptr, mean_ptr, inv_std_ptr, input_ptr, output_grad_ptr, output_ptr, 
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
    col_modulo = xindex % 32
    input_grad = tl.load(input_grad_ptr + (row_index + 32 * col_index), xmask, other=0.0)
    mean = tl.load(mean_ptr + (row_index + 32 * col_index), xmask, other=0.0)
    inv_std = tl.load(inv_std_ptr + (row_index + 32 * col_modulo), xmask, eviction_policy='evict_last', other=0.0)
    input_data = tl.load(input_ptr + (row_index + 32 * col_modulo), xmask, eviction_policy='evict_last', other=0.0)
    
    normalized_input = mean + inv_std
    negative_one = -1.0
    clamped_input = triton_helpers.maximum(normalized_input, negative_one)
    one = 1.0
    clipped_input = triton_helpers.minimum(clamped_input, one)
    threshold = 20.0
    is_clipped = clipped_input > threshold
    exp_clipped_input = tl.math.exp(clipped_input)
    log1p_exp_clipped_input = tl.extra.cuda.libdevice.log1p(exp_clipped_input)
    tanh_result = tl.where(is_clipped, clipped_input, log1p_exp_clipped_input)
    tanh_derivative = tl.extra.cuda.libdevice.tanh(tanh_result)
    derivative_product = clipped_input * tanh_derivative
    grad_input = input_grad * derivative_product
    grad_input_weighted = grad_input * input_data
    broadcast_grad_input_weighted = tl.broadcast_to(grad_input_weighted, [XBLOCK, RBLOCK])
    masked_grad_input_weighted = tl.where(xmask, broadcast_grad_input_weighted, 0)
    sum_grad_input_weighted = tl.sum(masked_grad_input_weighted, 1)[:, None]
    
    grad_input_unweighted = input_grad * input_data
    broadcast_grad_input_unweighted = tl.broadcast_to(grad_input_unweighted, [XBLOCK, RBLOCK])
    masked_grad_input_unweighted = tl.where(xmask, broadcast_grad_input_unweighted, 0)
    sum_grad_input_unweighted = tl.sum(masked_grad_input_unweighted, 1)[:, None]
    
    tl.store(output_grad_ptr + (col_index), sum_grad_input_weighted, xmask)
    tl.store(output_ptr + (col_index), sum_grad_input_unweighted, xmask)