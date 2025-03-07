# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_293(
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
    mean_value = tl.load(input_mean_ptr + (row_index + 49 * x0 + 78400 * channel_index), rmask, other=0.0)
    var_value = tl.load(input_var_ptr + (x0), None, eviction_policy='evict_last')
    data_value = tl.load(input_data_ptr + (x0), None, eviction_policy='evict_last')
    
    mean_broadcast = tl.broadcast_to(mean_value, [RBLOCK])
    mean_adjusted = tl.where(rmask, mean_broadcast, 0)
    mean_sum = triton_helpers.promote_to_tensor(tl.sum(mean_adjusted, 0))
    mean_count = tl.full([1], 490, tl.int32).to(tl.float32)
    mean_avg = mean_sum / mean_count
    
    mean_diff = mean_broadcast - mean_avg
    mean_diff_squared = mean_diff * mean_diff
    var_broadcast = tl.broadcast_to(mean_diff_squared, [RBLOCK])
    var_adjusted = tl.where(rmask, var_broadcast, 0)
    var_sum = triton_helpers.promote_to_tensor(tl.sum(var_adjusted, 0))
    var_avg = var_sum / 490.0
    
    epsilon = 1e-05
    var_stabilized = var_avg + epsilon
    inv_var = tl.extra.cuda.libdevice.rsqrt(var_stabilized)
    
    momentum = 0.1
    running_mean = mean_avg * momentum
    running_mean_update = running_mean + var_value * (1 - momentum)
    
    variance_momentum = 1.0020449897750512
    running_var = var_avg * variance_momentum
    running_var_update = running_var * momentum + var_value * (1 - momentum)
    
    tl.store(output_normalized_ptr + (x0), inv_var, None)
    tl.store(output_mean_ptr + (x0), running_mean_update, None)
    tl.store(output_var_ptr + (x0), running_var_update, None)
    tl.store(output_scale_ptr + (x0), mean_avg, None)
    tl.store(output_bias_ptr + (x0), var_avg, None)