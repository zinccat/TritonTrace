# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_289(
    input_mean_ptr, input_var_ptr, input_data_ptr, 
    output_normalized_ptr, output_mean_ptr, output_var_ptr, 
    output_scale_ptr, output_bias_ptr, xnumel, rnumel
):
    XBLOCK: tl.constexpr = 1
    rnumel = 490
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    row_index = rindex % 49
    channel_index = rindex // 49
    x0 = xindex
    mean_value = tl.load(input_mean_ptr + (row_index + 49 * x0 + 75264 * channel_index), rmask, other=0.0)
    var_value = tl.load(input_var_ptr + (x0), None, eviction_policy='evict_last')
    data_value = tl.load(input_data_ptr + (x0), None, eviction_policy='evict_last')
    
    mean_broadcast = tl.broadcast_to(mean_value, [RBLOCK])
    mean_adjusted = tl.where(rmask, mean_broadcast, 0)
    mean_adjusted_sum = triton_helpers.promote_to_tensor(tl.sum(mean_adjusted, 0))
    mean_adjusted_sum_div = mean_adjusted_sum / tl.full([1], 490, tl.int32).to(tl.float32)
    mean_centered = mean_broadcast - mean_adjusted_sum_div
    variance = mean_centered * mean_centered
    variance_broadcast = tl.broadcast_to(variance, [RBLOCK])
    variance_adjusted = tl.where(rmask, variance_broadcast, 0)
    variance_sum = triton_helpers.promote_to_tensor(tl.sum(variance_adjusted, 0))
    variance_mean = variance_sum / 490.0
    epsilon = 1e-05
    variance_stabilized = variance_mean + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_stabilized)
    
    momentum = 0.1
    running_mean = mean_adjusted_sum_div * momentum
    running_mean_update = running_mean + var_value * 0.9
    running_var_update = variance_mean * 1.0020449897750512 * momentum + var_value * 0.9
    
    tl.store(output_normalized_ptr + (x0), inv_stddev, None)
    tl.store(output_mean_ptr + (x0), running_mean_update, None)
    tl.store(output_var_ptr + (x0), running_var_update, None)
    tl.store(output_scale_ptr + (x0), mean_adjusted_sum_div, None)
    tl.store(output_bias_ptr + (x0), variance_sum, None)