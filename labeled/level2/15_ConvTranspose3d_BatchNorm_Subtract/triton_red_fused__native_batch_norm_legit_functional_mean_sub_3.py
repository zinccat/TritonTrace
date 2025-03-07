# From: 15_ConvTranspose3d_BatchNorm_Subtract

import triton
import triton.language as tl


@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_mean_sub_3(
    input_ptr_mean, input_ptr_var, input_ptr_rmean, input_ptr_rvar, input_ptr_gamma, 
    output_ptr_sub, output_ptr_norm, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 512
    rnumel = 123039
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 32
    loaded_mean = tl.load(input_ptr_mean + (x0), xmask, eviction_policy='evict_last')
    loaded_var = tl.load(input_ptr_var + (x0), xmask, eviction_policy='evict_last')
    loaded_rmean = tl.load(input_ptr_rmean + (x0), xmask, eviction_policy='evict_last')
    loaded_rvar = tl.load(input_ptr_rvar + (x0), xmask, eviction_policy='evict_last')
    accumulated_result = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r2 = rindex % 3969
        r3 = (rindex // 3969)
        loaded_input = tl.load(input_ptr_gamma + (r5 + (123039 * x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        normalized_input = loaded_input - loaded_mean
        variance_scale = 1968624.0
        epsilon = 1e-05
        normalized_variance = loaded_var / variance_scale
        adjusted_variance = normalized_variance + epsilon
        inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
        scaled_input = normalized_input * inv_sqrt_variance
        gamma_scaled_input = scaled_input * loaded_rmean
        output = gamma_scaled_input + loaded_rvar
        broadcast_output = tl.broadcast_to(output, [XBLOCK, RBLOCK])
        accumulated_result = accumulated_result + broadcast_output
        accumulated_result = tl.where(rmask & xmask, accumulated_result, accumulated_result)
        tl.store(output_ptr_sub + (r2 + (4000 * r3) + (124000 * x4)), output, rmask & xmask)
    
    sum_accumulated_result = tl.sum(accumulated_result, 1)[:, None]
    
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex % 3969
        r3 = (rindex // 3969)
        r5 = rindex
        loaded_output = tl.load(output_ptr_sub + (r2 + (4000 * r3) + (124000 * x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        mean_count = 123039.0
        mean_adjustment = sum_accumulated_result / mean_count
        adjusted_output = loaded_output - mean_adjustment
        tl.store(output_ptr_norm + (r5 + (123039 * x4)), adjusted_output, rmask & xmask)