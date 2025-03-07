# From: 34_ConvTranspose3d_LayerNorm_GELU_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_convolution_gelu_mul_native_layer_norm_0(
    in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, out_ptr0, kernel_size, xnumel, rnumel, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = r_index
    x4 = x_index
    x1 = ((x_index // kernel_size) % 64)
    
    # Load data
    input_data = tl.load(in_out_ptr0 + (r3 + 64 * x4), None)
    kernel_data = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    layer_norm_data1 = tl.load(in_ptr1 + (r3), None, eviction_policy='evict_last')
    layer_norm_data2 = tl.load(in_ptr2 + (r3), None, eviction_policy='evict_last')
    
    # Compute intermediate values
    sum_data = input_data + kernel_data
    broadcast_sum = tl.broadcast_to(sum_data, [XBLOCK, RBLOCK])
    sum_over_rblock = tl.sum(broadcast_sum, 1)[:, None]
    mean = sum_over_rblock / tl.full([XBLOCK, 1], 64, tl.int32).to(tl.float32)
    centered_data = broadcast_sum - mean
    squared_data = centered_data * centered_data
    broadcast_squared = tl.broadcast_to(squared_data, [XBLOCK, RBLOCK])
    sum_squared = tl.sum(broadcast_squared, 1)[:, None]
    variance = sum_squared / 64.0
    epsilon = 1e-05
    variance_with_epsilon = variance + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    normalized_data = centered_data * inv_stddev
    
    # Apply layer normalization
    layer_norm_output = normalized_data * layer_norm_data1 + layer_norm_data2
    
    # Apply GELU activation
    half = 0.5
    sqrt_2_over_pi = 0.7071067811865476
    gelu_input = layer_norm_output * sqrt_2_over_pi
    erf_result = tl.extra.cuda.libdevice.erf(gelu_input)
    gelu_output = half * layer_norm_output * (erf_result + 1.0)
    
    # Store results
    tl.store(in_out_ptr0 + (r3 + 64 * x4), input_data, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), inv_stddev, None)
    tl.store(in_out_ptr2 + (r3 + 64 * x4), gelu_output, None)
    tl.store(out_ptr0 + (x4), mean, None)