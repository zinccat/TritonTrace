# From: 58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_clamp_div_logsumexp_max_mul_sigmoid_sub_2(
    in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    
    input_indices = xindex
    reduction_indices = rindex
    
    loaded_in_out = tl.load(in_out_ptr0 + (input_indices), xmask, eviction_policy='evict_last')
    loaded_in0 = tl.load(in_ptr0 + (input_indices), xmask, eviction_policy='evict_last')
    loaded_in1 = tl.load(in_ptr1 + (reduction_indices), None, eviction_policy='evict_last')
    
    log_loaded_in_out = tl.math.log(loaded_in_out)
    abs_loaded_in0 = tl.math.abs(loaded_in0)
    inf_value = float("inf")
    is_inf = abs_loaded_in0 == inf_value
    zero_value = 0.0
    clamped_in0 = tl.where(is_inf, zero_value, loaded_in0)
    
    log_plus_clamped = log_loaded_in_out + clamped_in0
    bias = 3.0
    biased_log = log_plus_clamped + bias
    sigmoid_result = tl.sigmoid(biased_log)
    log_times_sigmoid = log_plus_clamped * sigmoid_result
    scaling_factor = 0.16666666666666666
    scaled_result = log_times_sigmoid * scaling_factor
    
    subtracted_result = scaled_result - loaded_in1
    clamp_min = -1.0
    clamped_result = triton_helpers.maximum(subtracted_result, clamp_min)
    clamp_max = 1.0
    clamped_and_bounded = triton_helpers.minimum(clamped_result, clamp_max)
    
    broadcast_clamped = tl.broadcast_to(clamped_and_bounded, [XBLOCK, RBLOCK])
    masked_clamped = tl.where(xmask, broadcast_clamped, float("-inf"))
    
    max_values, _ = triton_helpers.max2(masked_clamped, 1)[:, None]
    broadcast_rindex = tl.broadcast_to(reduction_indices, masked_clamped.shape)
    max_values_with_indices, max_indices = triton_helpers.max_with_index(masked_clamped, broadcast_rindex, 1)
    
    tl.debug_barrier()
    
    tl.store(in_out_ptr0 + (input_indices), log_plus_clamped, xmask)
    tl.store(out_ptr0 + (input_indices), max_values, xmask)
    tl.store(out_ptr1 + (input_indices), max_indices, xmask)