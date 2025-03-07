# From: 97_Matmul_BatchNorm_BiasAdd_Divide_Swish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_div_mul_sigmoid_1(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    
    # Load input data
    input_data = tl.load(in_ptr0 + (x2), xmask)
    mean = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    variance = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    gamma = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    beta = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    bias = tl.load(in_ptr5 + (0))
    
    # Broadcast bias
    broadcast_bias = tl.broadcast_to(bias, [XBLOCK])
    
    # Batch normalization computation
    normalized_data = input_data - mean
    scaled_data = normalized_data * variance
    scaled_gamma = scaled_data * gamma
    shifted_data = scaled_gamma + beta
    output_data = shifted_data + broadcast_bias
    
    # Swish activation
    swish_input = output_data
    swish_output = swish_input * tl.sigmoid(swish_input)
    
    # Store result
    tl.store(in_out_ptr0 + (x2), swish_output, xmask)