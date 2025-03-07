# From: 95_Matmul_Add_Swish_Tanh_GELU_Hardtanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_gelu_hardtanh_mul_sigmoid_tanh_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    
    input_val0 = tl.load(in_ptr0 + (x2), xmask)
    input_val1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    
    add_result = input_val0 + input_val1
    sigmoid_result = tl.sigmoid(add_result)
    mul_result = sigmoid_result * add_result
    tanh_result = tl.extra.cuda.libdevice.tanh(mul_result)
    
    half = 0.5
    scaled_tanh = tanh_result * half
    
    sqrt_half = 0.7071067811865476
    scaled_sqrt_half_tanh = tanh_result * sqrt_half
    erf_result = tl.extra.cuda.libdevice.erf(scaled_sqrt_half_tanh)
    
    one = 1.0
    erf_plus_one = erf_result + one
    
    gelu_result = scaled_tanh * erf_plus_one
    
    negative_one = -1.0
    hardtanh_result = triton_helpers.maximum(gelu_result, negative_one)
    clamped_result = triton_helpers.minimum(hardtanh_result, one)
    
    tl.store(out_ptr0 + (x2), clamped_result, xmask)