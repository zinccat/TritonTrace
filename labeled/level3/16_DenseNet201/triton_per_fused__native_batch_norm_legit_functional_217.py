# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_217(
    input_mean_ptr, input_var_ptr, input_data_ptr, 
    output_normalized_ptr, output_mean_ptr, output_var_ptr, 
    output_scale_ptr, output_bias_ptr, xnumel, rnumel):

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

    # Load input mean
    input_mean = tl.load(input_mean_ptr + (row_index + 49 * x0 + 43904 * channel_index), rmask, other=0.0)
    input_var = tl.load(input_var_ptr + (x0), None, eviction_policy='evict_last')
    input_data = tl.load(input_data_ptr + (x0), None, eviction_policy='evict_last')

    # Broadcast input mean
    broadcast_mean = tl.broadcast_to(input_mean, [RBLOCK])
    masked_mean = tl.where(rmask, broadcast_mean, 0)

    # Calculate mean
    sum_mean = triton_helpers.promote_to_tensor(tl.sum(masked_mean, 0))
    num_elements = tl.full([1], 490, tl.int32).to(tl.float32)
    mean = sum_mean / num_elements

    # Calculate variance
    mean_diff = broadcast_mean - mean
    squared_diff = mean_diff * mean_diff
    broadcast_squared_diff = tl.broadcast_to(squared_diff, [RBLOCK])
    masked_squared_diff = tl.where(rmask, broadcast_squared_diff, 0)
    sum_squared_diff = triton_helpers.promote_to_tensor(tl.sum(masked_squared_diff, 0))
    variance = sum_squared_diff / 490.0

    # Calculate scale and bias
    epsilon = 1e-05
    inv_std = tl.extra.cuda.libdevice.rsqrt(variance + epsilon)
    momentum = 0.1
    running_mean = mean * momentum
    running_var = variance * 1.0020449897750512 * momentum

    # Store results
    tl.store(output_normalized_ptr + (x0), inv_std, None)
    tl.store(output_mean_ptr + (x0), running_mean, None)
    tl.store(output_var_ptr + (x0), running_var, None)
    tl.store(output_scale_ptr + (x0), mean, None)
    tl.store(output_bias_ptr + (x0), variance, None)