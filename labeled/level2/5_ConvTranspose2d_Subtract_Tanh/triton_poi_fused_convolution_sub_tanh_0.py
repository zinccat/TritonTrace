# From: 5_ConvTranspose2d_Subtract_Tanh

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_convolution_sub_tanh_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    x3 = x_index
    x1 = (x_index // 1089) % 16
    
    output_value = tl.load(in_out_ptr0 + (x3), None)
    input_value_0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    input_value_1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    
    add_result = output_value + input_value_0
    subtract_result = add_result - input_value_1
    
    tanh_result = tl.extra.cuda.libdevice.tanh(subtract_result)
    
    tl.store(in_out_ptr0 + (x3), tanh_result, None)