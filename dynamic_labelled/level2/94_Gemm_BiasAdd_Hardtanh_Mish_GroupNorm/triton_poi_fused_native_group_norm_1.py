# From: 94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_1(
    input_ptr_mean, input_ptr_var, input_ptr_bn_mean, input_ptr_bn_var, input_ptr_bn_weight, input_ptr_bn_bias, 
    output_ptr, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    element_index = index
    block_index = index % 1024

    mean = tl.load(input_ptr_mean + (element_index), mask)
    var = tl.load(input_ptr_var + (block_index), mask, eviction_policy='evict_last')
    bn_mean = tl.load(input_ptr_bn_mean + (element_index // 32), mask, eviction_policy='evict_last')
    bn_var = tl.load(input_ptr_bn_var + (element_index // 32), mask, eviction_policy='evict_last')
    bn_weight = tl.load(input_ptr_bn_weight + (block_index), mask, eviction_policy='evict_last')
    bn_bias = tl.load(input_ptr_bn_bias + (block_index), mask, eviction_policy='evict_last')

    normalized = mean + var
    relu_min = -1.0
    relu_max = 1.0
    relu_clipped = triton_helpers.maximum(normalized, relu_min)
    relu_clipped = triton_helpers.minimum(relu_clipped, relu_max)

    mish_threshold = 20.0
    mish_exceeds_threshold = relu_clipped > mish_threshold
    exp_relu_clipped = tl.math.exp(relu_clipped)
    log1p_exp_relu_clipped = tl.extra.cuda.libdevice.log1p(exp_relu_clipped)
    mish_activation = tl.where(mish_exceeds_threshold, relu_clipped, log1p_exp_relu_clipped)
    tanh_mish = tl.extra.cuda.libdevice.tanh(mish_activation)
    mish_output = relu_clipped * tanh_mish

    normalized_output = mish_output - bn_mean
    bn_var_scaled = bn_var / 32.0
    epsilon = 1e-05
    variance_inverse_sqrt = tl.extra.cuda.libdevice.rsqrt(bn_var_scaled + epsilon)
    normalized_scaled = normalized_output * variance_inverse_sqrt
    scaled_output = normalized_scaled * bn_weight
    final_output = scaled_output + bn_bias

    tl.store(output_ptr + (element_index), final_output, mask)