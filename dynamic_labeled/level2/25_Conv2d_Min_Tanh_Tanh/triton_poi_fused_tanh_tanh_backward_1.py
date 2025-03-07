# From: 25_Conv2d_Min_Tanh_Tanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_tanh_tanh_backward_1(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    # Load input values
    input_values = tl.load(in_ptr0 + (x0), xmask)

    # Apply tanh function twice
    tanh_output1 = tl.extra.cuda.libdevice.tanh(input_values)
    tanh_output2 = tl.extra.cuda.libdevice.tanh(tanh_output1)

    # Calculate derivative of tanh
    tanh_derivative = tanh_output1 * tanh_output1
    derivative_output = 1.0 - tanh_derivative

    # Store results
    tl.store(out_ptr0 + (x0), tanh_output2, xmask)
    tl.store(out_ptr1 + (x0), derivative_output, xmask)