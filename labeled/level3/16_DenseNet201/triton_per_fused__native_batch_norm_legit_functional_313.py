# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_313(
    input_mean_ptr, input_var_ptr, input_data_ptr, 
    output_mean_ptr, output_var_ptr, output_data_ptr, 
    xnumel, rnumel
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
    mean_value = tl.load(input_mean_ptr + (row_index + 49 * x0 + 94080 * channel_index), rmask, other=0.0)
    var_value = tl.load(input_var_ptr + (x0), None, eviction_policy='evict_last')
    data_value = tl.load(input_data_ptr + (x0), None, eviction_policy='evict_last')
    
    broadcast_mean = tl.broadcast_to(mean_value, [RBLOCK])
    masked_mean = tl.where(rmask, broadcast_mean, 0)
    mean_sum = triton_helpers.promote_to_tensor(tl.sum(masked_mean, 0))
    num_elements = tl.full([1], 490, tl.int32)
    num_elements_float = num_elements.to(tl.float32)
    mean = mean_sum / num_elements_float
    
    mean_centered = mean_value - mean
    squared_diff = mean_centered * mean_centered
    broadcast_squared_diff = tl.broadcast_to(squared_diff, [RBLOCK])
    masked_squared_diff = tl.where(rmask, broadcast_squared_diff, 0)
    var_sum = triton_helpers.promote_to_tensor(tl.sum(masked_squared_diff, 0))
    variance = var_sum / 490.0
    epsilon = 1e-05
    variance_eps = variance + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(variance_eps)
    
    momentum = 0.1
    running_mean = mean * momentum
    running_mean_factor = 1.0020449897750512
    running_mean_update = variance * running_mean_factor * momentum
    running_mean_final = running_mean + running_mean_update
    
    momentum_var = 0.9
    running_var = var_value * momentum_var
    running_var_final = running_mean_update + running_var
    
    tl.store(output_mean_ptr + (x0), inv_std, None)
    tl.store(output_var_ptr + (x0), running_mean_final, None)
    tl.store(output_data_ptr + (x0), running_var_final, None)
    tl.store(output_mean_ptr + (x0), mean, None)