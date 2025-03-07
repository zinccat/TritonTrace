# From: 34_ConvTranspose3d_LayerNorm_GELU_Scaling

import triton
import triton.language as tl


@triton.jit
def triton_per_fused_convolution_gelu_mul_native_layer_norm_0(
    in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, out_ptr0, 
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    row_indices = rindex
    col_indices = xindex
    batch_index = (xindex // 2048) % 64
    input_value = tl.load(in_out_ptr0 + (row_indices + (64 * col_indices)), None)
    weight_value = tl.load(in_ptr0 + (batch_index), None, eviction_policy='evict_last')
    layer_norm_mean = tl.load(in_ptr1 + (row_indices), None, eviction_policy='evict_last')
    layer_norm_var = tl.load(in_ptr2 + (row_indices), None, eviction_policy='evict_last')
    
    sum_input_weight = input_value + weight_value
    broadcast_sum = tl.broadcast_to(sum_input_weight, [XBLOCK, RBLOCK])
    sum_broadcast = tl.sum(broadcast_sum, 1)[:, None]
    num_elements = tl.full([XBLOCK, 1], 64, tl.int32).to(tl.float32)
    mean = sum_broadcast / num_elements
    
    centered_values = broadcast_sum - mean
    squared_values = centered_values * centered_values
    broadcast_squared = tl.broadcast_to(squared_values, [XBLOCK, RBLOCK])
    sum_squared = tl.sum(broadcast_squared, 1)[:, None]
    variance = sum_squared / 64.0
    epsilon = 1e-05
    variance_epsilon = variance + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_epsilon)
    
    normalized_values = centered_values * inv_stddev
    layer_norm_output = normalized_values * layer_norm_mean + layer_norm_var
    
    gelu_input = layer_norm_output * 0.5
    sqrt_2_over_pi = 0.7071067811865476
    erf_input = layer_norm_output * sqrt_2_over_pi
    erf_output = tl.extra.cuda.libdevice.erf(erf_input)
    gelu_output = gelu_input * (erf_output + 1.0) * 1.0
    
    tl.store(in_out_ptr0 + (row_indices + (64 * col_indices)), sum_input_weight, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (col_indices), inv_stddev, None)
    tl.store(in_out_ptr2 + (row_indices + (64 * col_indices)), gelu_output, None)
    tl.store(out_ptr0 + (col_indices), mean, None)