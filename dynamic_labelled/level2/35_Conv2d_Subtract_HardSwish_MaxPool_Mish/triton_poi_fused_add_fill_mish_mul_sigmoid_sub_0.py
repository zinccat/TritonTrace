# From: 35_Conv2d_Subtract_HardSwish_MaxPool_Mish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_fill_mish_mul_sigmoid_sub_0poi_fused_add_fill_mish_mul_sigmoid_sub_0(
    in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    input_value = tl.load(in_ptr0 + (x0), xmask)
    output_value = tl.load(in_out_ptr0 + (x0), xmask)

    threshold = 20.0
    is_greater_than_threshold = output_value > threshold

    exp_output = tl.math.exp(output_value)
    log1p_exp_output = tl.extra.cuda.libdevice.log1p(exp_output)

    log_output = tl.where(is_greater_than_threshold, output_value, log1p_exp_output)
    tanh_log_output = tl.extra.cuda.libdevice.tanh(log_output)

    sigmoid_output = tl.sigmoid(output_value)
    input_times_sigmoid = output_value * sigmoid_output

    tanh_squared = tanh_log_output * tanh_log_output
    one_minus_tanh_squared = 1.0 - tanh_squared

    mish_output = input_times_sigmoid * one_minus_tanh_squared
    final_output = tanh_log_output + mish_output

    result = input_value * final_output

    tl.store(in_out_ptr0 + (x0), result, xmask)