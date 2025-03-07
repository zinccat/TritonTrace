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
    r_block = r_index
    x_block = x_index
    x_channel = ((x_index // kernel_size) % 64)
    
    input_value0 = tl.load(in_out_ptr0 + (r_block + 64 * x_block), None)
    input_value1 = tl.load(in_ptr0 + (x_channel), None, eviction_policy='evict_last')
    input_value2 = tl.load(in_ptr1 + (r_block), None, eviction_policy='evict_last')
    input_value3 = tl.load(in_ptr2 + (r_block), None, eviction_policy='evict_last')
    
    sum_value = input_value0 + input_value1
    broadcast_sum = tl.broadcast_to(sum_value, [XBLOCK, RBLOCK])
    sum_across_r = tl.sum(broadcast_sum, 1)[:, None]
    num_elements = tl.full([XBLOCK, 1], 64, tl.int32).to(tl.float32)
    mean_value = sum_across_r / num_elements
    
    centered_value = broadcast_sum - mean_value
    squared_value = centered_value * centered_value
    broadcast_squared = tl.broadcast_to(squared_value, [XBLOCK, RBLOCK])
    sum_squared = tl.sum(broadcast_squared, 1)[:, None]
    variance = sum_squared / 64.0
    epsilon = 1e-05
    variance_with_epsilon = variance + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    
    normalized_value = centered_value * inv_stddev
    scaled_value = normalized_value * input_value2
    layer_norm_output = scaled_value + input_value3
    
    gelu_input = layer_norm_output * 0.5
    sqrt_2_over_pi = 0.7071067811865476
    erf_input = layer_norm_output * sqrt_2_over_pi
    erf_output = tl.extra.cuda.libdevice.erf(erf_input)
    gelu_output = gelu_input * (erf_output + 1.0) * 1.0
    
    tl.store(in_out_ptr0 + (r_block + 64 * x_block), sum_value, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x_block), inv_stddev, None)
    tl.store(in_out_ptr2 + (r_block + 64 * x_block), gelu_output, None)
    tl.store(out_ptr0 + (x_block), mean_value, None)