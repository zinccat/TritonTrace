# From: 82_Conv2d_Tanh_Scaling_BiasAdd_Max

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_add_convolution_mul_tanh_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    x3 = x_index
    x1 = (x_index // 900) % 16
    
    input_out_value = tl.load(in_out_ptr0 + (x3), None)
    input_value_0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    input_value_1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    
    add_result = input_out_value + input_value_0
    tanh_result = tl.extra.cuda.libdevice.tanh(add_result)
    
    scale_factor = 2.0
    scaled_tanh = tanh_result * scale_factor
    
    final_result = scaled_tanh + input_value_1
    
    tl.store(in_out_ptr0 + (x3), add_result, None)
    tl.store(out_ptr0 + (x3), final_result, None)