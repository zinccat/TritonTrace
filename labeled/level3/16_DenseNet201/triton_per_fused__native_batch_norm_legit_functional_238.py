# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_238(
    input_mean_ptr, input_var_ptr, input_data_ptr, 
    output_mean_ptr, output_var_ptr, output_data_ptr, 
    output_rmean_ptr, output_rvar_ptr, xnumel, rnumel):

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
    input_mean = tl.load(input_mean_ptr + (row_index + 49 * x0 + 50176 * channel_index), rmask, other=0.0)
    # Load input variance
    input_variance = tl.load(input_var_ptr + (x0), None, eviction_policy='evict_last')
    # Load input data
    input_data = tl.load(input_data_ptr + (x0), None, eviction_policy='evict_last')

    # Broadcast input mean
    broadcast_mean = tl.broadcast_to(input_mean, [RBLOCK])
    masked_mean = tl.where(rmask, broadcast_mean, 0)

    # Calculate mean
    sum_mean = triton_helpers.promote_to_tensor(tl.sum(masked_mean, 0))
    num_elements = tl.full([1], 490, tl.int32).to(tl.float32)
    calculated_mean = sum_mean / num_elements

    # Calculate variance
    mean_diff = broadcast_mean - calculated_mean
    squared_diff = mean_diff * mean_diff
    broadcast_squared_diff = tl.broadcast_to(squared_diff, [RBLOCK])
    masked_squared_diff = tl.where(rmask, broadcast_squared_diff, 0)
    sum_squared_diff = triton_helpers.promote_to_tensor(tl.sum(masked_squared_diff, 0))
    calculated_variance = sum_squared_diff / 490.0

    # Calculate running mean and variance
    epsilon = 1e-05
    adjusted_variance = calculated_variance + epsilon
    reciprocal_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    momentum = 0.1
    running_mean = calculated_mean * momentum
    running_mean_factor = 0.9
    updated_running_mean = running_mean + input_variance * running_mean_factor
    variance_scale = 1.0020449897750512
    scaled_variance = calculated_variance * variance_scale
    running_variance = scaled_variance * momentum
    running_variance_factor = 0.9
    updated_running_variance = running_variance + input_variance * running_variance_factor

    # Store results
    tl.store(output_rmean_ptr + (x0), reciprocal_sqrt_variance, None)
    tl.store(output_rvar_ptr + (x0), updated_running_mean, None)
    tl.store(output_data_ptr + (x0), updated_running_variance, None)
    tl.store(output_mean_ptr + (x0), calculated_mean, None)
    tl.store(output_var_ptr + (x0), calculated_variance, None)