# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_154(
    input_ptr_mean, input_ptr_var, input_ptr_input, 
    output_ptr_normalized, output_ptr_mean, output_ptr_var, 
    output_ptr_inv_std, output_ptr_running_mean, output_ptr_running_var, 
    num_elements, num_features
):
    XBLOCK: tl.constexpr = 1
    num_features = 490
    RBLOCK: tl.constexpr = 512
    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[:]
    r_mask = r_index < num_features
    feature_index = r_index % 49
    batch_index = r_index // 49
    x_index_0 = x_index
    mean_value = tl.load(input_ptr_mean + (feature_index + 49 * x_index_0 + 28224 * batch_index), r_mask, other=0.0)
    running_mean_value = tl.load(input_ptr_var + (x_index_0), None, eviction_policy='evict_last')
    running_var_value = tl.load(input_ptr_input + (x_index_0), None, eviction_policy='evict_last')
    broadcast_mean = tl.broadcast_to(mean_value, [RBLOCK])
    masked_broadcast_mean = tl.where(r_mask, broadcast_mean, 0)
    squared_diff = (broadcast_mean - tl.broadcast_to(tl.sum(masked_broadcast_mean, 0) / tl.full([1], num_features, tl.int32).to(tl.float32))) ** 2
    masked_squared_diff = tl.where(r_mask, squared_diff, 0)
    variance = tl.sum(masked_squared_diff, 0) / 490.0
    epsilon = 1e-05
    inv_std = tl.extra.cuda.libdevice.rsqrt(variance + epsilon)
    momentum = 0.1
    updated_running_mean = running_mean_value * 0.9 + tl.sum(masked_broadcast_mean, 0) / 490.0 * momentum
    variance_scale = 1.0020449897750512
    updated_running_var = running_var_value * 0.9 + variance * variance_scale * momentum
    normalized_value = (broadcast_mean - tl.broadcast_to(tl.sum(masked_broadcast_mean, 0) / tl.full([1], num_features, tl.int32).to(tl.float32))) * inv_std

    tl.store(output_ptr_normalized + (x_index_0), normalized_value, None)
    tl.store(output_ptr_mean + (x_index_0), tl.sum(masked_broadcast_mean, 0) / tl.full([1], num_features, tl.int32).to(tl.float32), None)
    tl.store(output_ptr_var + (x_index_0), variance, None)
    tl.store(output_ptr_inv_std + (x_index_0), inv_std, None)
    tl.store(output_ptr_running_mean + (x_index_0), updated_running_mean, None)
    tl.store(output_ptr_running_var + (x_index_0), updated_running_var, None)