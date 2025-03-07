# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_189(
    input_mean_ptr, input_var_ptr, input_data_ptr, 
    output_mean_ptr, output_var_ptr, output_data_ptr, 
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
    mean_value = tl.load(input_mean_ptr + (row_index + 49 * x0 + 47040 * channel_index), rmask, other=0.0)
    var_value = tl.load(input_var_ptr + (x0), None, eviction_policy='evict_last')
    data_value = tl.load(input_data_ptr + (x0), None, eviction_policy='evict_last')
    
    mean_broadcast = tl.broadcast_to(mean_value, [RBLOCK])
    mean_adjusted = tl.where(rmask, mean_broadcast, 0)
    mean_sum = triton_helpers.promote_to_tensor(tl.sum(mean_adjusted, 0))
    mean_count = tl.full([1], 490, tl.int32).to(tl.float32)
    mean_normalized = mean_sum / mean_count
    
    mean_centered = mean_broadcast - mean_normalized
    variance = mean_centered * mean_centered
    variance_broadcast = tl.broadcast_to(variance, [RBLOCK])
    variance_adjusted = tl.where(rmask, variance_broadcast, 0)
    variance_sum = triton_helpers.promote_to_tensor(tl.sum(variance_adjusted, 0))
    variance_normalized = variance_sum / 490.0
    epsilon = 1e-05
    variance_stabilized = variance_normalized + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_stabilized)
    
    momentum = 0.1
    mean_momentum = mean_normalized * momentum
    mean_update = tl.load(output_mean_ptr + (x0), None, eviction_policy='evict_last') * 0.9
    updated_mean = mean_momentum + mean_update
    
    variance_momentum = variance_stabilized * 1.0020449897750512 * momentum
    variance_update = tl.load(output_var_ptr + (x0), None, eviction_policy='evict_last') * 0.9
    updated_variance = variance_momentum + variance_update
    
    tl.store(output_scale_ptr + (x0), inv_stddev, None)
    tl.store(output_bias_ptr + (x0), updated_mean, None)
    tl.store(output_var_ptr + (x0), updated_variance, None)
    tl.store(output_mean_ptr + (x0), mean_normalized, None)
    tl.store(output_data_ptr + (x0), variance_sum, None)