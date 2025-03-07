# From: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_tanh_3(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_input, 
    output_ptr, num_elements, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    x3 = x_index
    x1 = (x_index // 4096) % 64
    
    mean = tl.load(input_ptr_mean + (x3), None)
    var = tl.load(input_ptr_var + (x1), None, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (x1), None, eviction_policy='evict_last')
    bias = tl.load(input_ptr_bias + (x1), None, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (x3), None)
    
    normalized_input = input_data - mean
    inv_std_dev = tl.extra.cuda.libdevice.rsqrt(var / 524288.0 + 1e-05)
    scaled_input = normalized_input * inv_std_dev * scale
    biased_input = scaled_input + bias
    tanh_output = tl.extra.cuda.libdevice.tanh(biased_input)
    
    tl.store(output_ptr + (x3), tanh_output, None)