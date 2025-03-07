# From: 7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_add_convolution_gelu_leaky_relu_relu_sigmoid_0(
    in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 12600) % 16

    # Load data from pointers
    input_output_data = tl.load(in_out_ptr0 + (x3), None)
    input_data_0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    input_data_1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')

    # Perform addition
    added_data = input_output_data + input_data_0

    # Leaky ReLU
    zero_tensor = tl.full([1], 0, tl.int32)
    max_data = triton_helpers.maximum(zero_tensor, added_data)
    leaky_relu_slope = 0.01
    leaky_relu_data = tl.where(max_data > 0, max_data, max_data * leaky_relu_slope)

    # GELU
    gelu_coefficient = 0.5
    gelu_sqrt_coefficient = 0.7071067811865476
    gelu_data = leaky_relu_data * gelu_coefficient
    erf_input = leaky_relu_data * gelu_sqrt_coefficient
    erf_result = tl.extra.cuda.libdevice.erf(erf_input)
    gelu_result = gelu_data * (erf_result + 1.0)

    # Sigmoid
    sigmoid_result = tl.sigmoid(gelu_result)

    # Final addition
    final_result = sigmoid_result + input_data_1

    # Store results
    tl.store(in_out_ptr0 + (x3), added_data, None)
    tl.store(out_ptr0 + (x3), final_result, None)