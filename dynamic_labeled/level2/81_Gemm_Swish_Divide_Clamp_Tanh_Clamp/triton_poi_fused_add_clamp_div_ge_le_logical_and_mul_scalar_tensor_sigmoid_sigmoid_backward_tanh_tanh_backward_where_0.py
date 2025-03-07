# From: 81_Gemm_Swish_Divide_Clamp_Tanh_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_clamp_div_ge_le_logical_and_mul_scalar_tensor_sigmoid_sigmoid_backward_tanh_tanh_backward_where_0(
    in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    # Load input data
    input_data = tl.load(in_out_ptr0 + (x0), xmask)
    additional_input = tl.load(in_ptr0 + (x0), xmask)

    # Sigmoid operation
    sigmoid_output = tl.sigmoid(input_data)
    sigmoid_derivative = input_data * sigmoid_output

    # Clamping values
    clamp_min = -1.0
    clamp_max = 1.0
    half = 0.5

    # Clamp sigmoid derivative
    is_within_clamp = (sigmoid_derivative * half >= clamp_min) & (sigmoid_derivative * half <= clamp_max)
    clamped_value = triton_helpers.maximum(sigmoid_derivative * half, clamp_min)
    clamped_value = triton_helpers.minimum(clamped_value, clamp_max)

    # Tanh operation
    tanh_output = tl.extra.cuda.libdevice.tanh(clamped_value)
    is_tanh_within_clamp = (tanh_output >= clamp_min) & (tanh_output <= clamp_max)

    # Conditional selection
    selected_input = tl.where(is_tanh_within_clamp, additional_input, 0.0)

    # Tanh backward operation
    tanh_squared = tanh_output * tanh_output
    tanh_derivative = clamp_max - tanh_squared
    tanh_backward_output = selected_input * tanh_derivative

    # Conditional selection for backward output
    backward_output = tl.where(is_within_clamp, tanh_backward_output, 0.0)

    # Final computation
    scaled_backward_output = backward_output * half
    scaled_sigmoid_output = scaled_backward_output * sigmoid_output
    scaled_input_data = scaled_backward_output * input_data
    sigmoid_derivative_scaled = sigmoid_output * (clamp_max - sigmoid_output)
    final_output = scaled_input_data * sigmoid_derivative_scaled
    final_output += scaled_sigmoid_output

    # Store result
    tl.store(in_out_ptr0 + (x0), final_output, xmask)