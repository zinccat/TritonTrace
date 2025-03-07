# From: 44_MiniGPTBlock

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_mul_pow_tanh_8poi_fused_add_mul_pow_tanh_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    input_indices = xindex
    input_value = tl.load(in_ptr0 + (input_indices), None)
    half = 0.5
    scaled_input = input_value * half
    squared_input = input_value * input_value
    cubed_input = squared_input * input_value
    tanh_coefficient = 0.044715
    tanh_adjustment = cubed_input * tanh_coefficient
    tanh_input = input_value + tanh_adjustment
    tanh_scale = 0.7978845608028654
    scaled_tanh_input = tanh_input * tanh_scale
    tanh_result = tl.extra.cuda.libdevice.tanh(scaled_tanh_input)
    one = 1.0
    tanh_offset = tanh_result + one
    output_value = scaled_input * tanh_offset
    tl.store(out_ptr0 + (input_indices), output_value, None)