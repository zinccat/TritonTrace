# From: 45_Gemm_Sigmoid_Sum_LogSumExp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_sigmoid_sigmoid_backward_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 20
    x2 = xindex

    # Load and broadcast the first input
    input_value = tl.load(in_ptr0 + (0))
    broadcasted_input = tl.broadcast_to(input_value, [XBLOCK])

    # Load the second input with masking
    masked_input = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')

    # Load the in-out pointer with masking
    in_out_value = tl.load(in_out_ptr0 + (x2), xmask)

    # Compute intermediate values
    product_input_masked = broadcasted_input * masked_input
    sigmoid_value = tl.sigmoid(in_out_value)
    one_minus_sigmoid = 1.0 - sigmoid_value
    gradient = sigmoid_value * one_minus_sigmoid

    # Compute the final result
    result = product_input_masked * gradient

    # Store the result back
    tl.store(in_out_ptr0 + (x2), result, xmask)