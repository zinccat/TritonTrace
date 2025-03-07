# From: 3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_convolution_native_layer_norm_0(
    in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, 
    out_ptr0, out_ptr1, kernel_size, xnumel, rnumel, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    row_indices = rindex
    col_indices = xindex
    batch_indices = ((xindex // kernel_size) % 64)
    
    input_value0 = tl.load(in_out_ptr0 + (row_indices + 64 * col_indices), None)
    input_value1 = tl.load(in_ptr0 + (batch_indices), None, eviction_policy='evict_last')
    bias_value = tl.load(in_ptr1 + (0))
    broadcast_bias = tl.broadcast_to(bias_value, [XBLOCK, RBLOCK])
    
    layer_norm_mean = tl.load(in_ptr2 + (row_indices), None, eviction_policy='evict_last')
    layer_norm_var = tl.load(in_ptr3 + (row_indices), None, eviction_policy='evict_last')
    
    sum_result = input_value0 + input_value1
    bias_added = sum_result + broadcast_bias
    broadcast_sum = tl.broadcast_to(bias_added, [XBLOCK, RBLOCK])
    
    sum_over_rows = tl.sum(broadcast_sum, 1)[:, None]
    num_elements = tl.full([XBLOCK, 1], 64, tl.int32).to(tl.float32)
    mean = sum_over_rows / num_elements
    
    centered_values = broadcast_sum - mean
    squared_values = centered_values * centered_values
    broadcast_squared = tl.broadcast_to(squared_values, [XBLOCK, RBLOCK])
    
    sum_squared = tl.sum(broadcast_squared, 1)[:, None]
    variance = sum_squared / 64.0
    epsilon = 1e-05
    variance_with_epsilon = variance + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    
    normalized_values = centered_values * inv_stddev
    layer_norm_output = normalized_values * layer_norm_mean + layer_norm_var
    
    tl.store(in_out_ptr0 + (row_indices + 64 * col_indices), sum_result, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (col_indices), inv_stddev, None)
    tl.store(out_ptr1 + (row_indices + 64 * col_indices), layer_norm_output, None)
    tl.store(out_ptr0 + (col_indices), mean, None)