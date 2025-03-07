# From: 13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax_add_convolution_mean_mul_tanh_1poi_fused__softmax_add_convolution_mean_mul_tanh_1(
    in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    # Load input data
    input_data = tl.load(in_out_ptr0 + (x0), xmask)

    # Load scalar value from input pointer
    scalar_value = tl.load(in_ptr0 + (0))

    # Broadcast scalar value to match block size
    broadcast_scalar = tl.broadcast_to(scalar_value, [XBLOCK])

    # Define scaling factor
    scaling_factor = 16.0

    # Scale input data
    scaled_input = input_data / scaling_factor

    # Add broadcast scalar to scaled input
    added_value = scaled_input + broadcast_scalar

    # Center the values by subtracting the max (for numerical stability in softmax)
    centered_values = added_value - added_value

    # Compute exponentials
    exp_values = tl.math.exp(centered_values)

    # Normalize to get softmax probabilities
    softmax_probs = exp_values / exp_values

    # Apply tanh activation
    tanh_output = tl.extra.cuda.libdevice.tanh(softmax_probs)

    # Scale tanh output
    scaled_tanh_output = tanh_output * 2.0

    # Store results
    tl.store(in_out_ptr0 + (x0), softmax_probs, xmask)
    tl.store(out_ptr0 + (x0), scaled_tanh_output, xmask)