# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_299(
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
    mean_value = tl.load(input_mean_ptr + (row_index + 49 * x0 + 83104 * channel_index), rmask, other=0.0)
    running_mean = tl.load(input_var_ptr + (x0), None, eviction_policy='evict_last')
    running_var = tl.load(input_data_ptr + (x0), None, eviction_policy='evict_last')
    mean_broadcast = tl.broadcast_to(mean_value, [RBLOCK])
    mean_adjusted = tl.where(rmask, mean_broadcast, 0)
    mean_diff = mean_broadcast - tl.broadcast_to(tl.sum(mean_adjusted, 0), [RBLOCK]) / 490.0
    variance = tl.where(rmask, mean_diff * mean_diff, 0)
    variance_sum = triton_helpers.promote_to_tensor(tl.sum(variance, 0))
    variance_normalized = variance_sum / 490.0
    epsilon = 1e-05
    inv_std = tl.extra.cuda.libdevice.rsqrt(variance_normalized + epsilon)
    momentum = 0.1
    updated_running_mean = running_mean * 0.9 + tl.broadcast_to(tl.sum(mean_diff, 0), [1]).to(tl.float32) * momentum
    scale_factor = 1.0020449897750512
    updated_running_var = variance_normalized * scale_factor * momentum
    updated_running_var += running_var * 0.9
    tl.store(output_normalized_ptr + (x0), inv_std, None)
    tl.store(output_mean_ptr + (x0), updated_running_mean, None)
    tl.store(output_var_ptr + (x0), updated_running_var, None)
    tl.store(output_scale_ptr + (x0), tl.broadcast_to(tl.sum(mean_diff, 0), [1]).to(tl.float32), None)
    tl.store(output_bias_ptr + (x0), variance_sum, None)