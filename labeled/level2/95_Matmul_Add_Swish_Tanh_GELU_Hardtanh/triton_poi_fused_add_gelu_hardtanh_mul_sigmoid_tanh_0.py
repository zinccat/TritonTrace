# From: 95_Matmul_Add_Swish_Tanh_GELU_Hardtanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_add_gelu_hardtanh_mul_sigmoid_tanh_0(input_ptr0, input_ptr1, output_ptr0, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    index2 = index
    index0 = index % 512
    input_val0 = tl.load(input_ptr0 + (index2), None)
    input_val1 = tl.load(input_ptr1 + (index0), None, eviction_policy='evict_last')
    sum_val = input_val0 + input_val1
    sigmoid_val = tl.sigmoid(sum_val)
    product_val = sigmoid_val * sum_val
    tanh_val = tl.extra.cuda.libdevice.tanh(product_val)
    half = 0.5
    half_tanh = tanh_val * half
    sqrt_half = 0.7071067811865476
    sqrt_half_tanh = tanh_val * sqrt_half
    erf_val = tl.extra.cuda.libdevice.erf(sqrt_half_tanh)
    one = 1.0
    erf_plus_one = erf_val + one
    gelu_val = half_tanh * erf_plus_one
    negative_one = -1.0
    max_val = triton_helpers.maximum(gelu_val, negative_one)
    hardtanh_val = triton_helpers.minimum(max_val, one)
    tl.store(output_ptr0 + (index2), hardtanh_val, None)