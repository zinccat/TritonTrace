# From: 39_Gemm_Scale_BatchNorm

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_mul_1(input_ptr, scale_ptr, bias_ptr, mean_ptr, inv_var_ptr, epsilon_ptr, output_ptr, num_elements, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    input_val = tl.load(input_ptr + (x2), None)
    scale_val = tl.load(scale_ptr + (x0), None, eviction_policy='evict_last')
    bias_val = tl.load(bias_ptr + (x0), None, eviction_policy='evict_last')
    mean_val = tl.load(mean_ptr + (x0), None, eviction_policy='evict_last')
    inv_var_val = tl.load(inv_var_ptr + (x0), None, eviction_policy='evict_last')
    epsilon_val = tl.load(epsilon_ptr + (x0), None, eviction_policy='evict_last')
    
    scaled_input = input_val * scale_val
    centered_input = scaled_input - mean_val
    inv_std_dev = 128.0
    inv_std_dev_scaled = inv_var_val / inv_std_dev
    epsilon_adjusted = inv_std_dev_scaled + 1e-05
    rsqrt_epsilon_adjusted = tl.extra.cuda.libdevice.rsqrt(epsilon_adjusted)
    normalized_input = centered_input * rsqrt_epsilon_adjusted
    output_val = normalized_input * inv_var_val + bias_val
    
    tl.store(output_ptr + (x2), output_val, None)