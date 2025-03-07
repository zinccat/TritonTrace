# From: 58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_div_exp_mul_sigmoid_sigmoid_backward_sub_5(in_out_ptr0, in_ptr0, in_ptr1, ks0, ks1, ks2, ks3, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < xnumel
    x_dim0 = x_index % ks0
    x_dim1 = x_index // ks1
    x_linear_index = x_index

    # Load data from in_ptr0
    in_ptr0_offset = (x_dim0 + ((-1) * x_dim1) + ((-4) * x_dim1 * ks3 * ks3) + 2 * ks2 * x_dim1 + 4 * ks3 * x_dim1 + ((-8) * ks2 * ks3 * x_dim1) + 8 * ks2 * x_dim1 * ks3 * ks3)
    tmp0 = tl.load(in_ptr0 + in_ptr0_offset, x_mask, eviction_policy='evict_last')

    # Load data from in_ptr1
    in_ptr1_offset = (x_dim0 + ((-1) * x_dim1) + ((-4) * x_dim1 * ks3 * ks3) + 2 * ks2 * x_dim1 + 4 * ks3 * x_dim1 + ((-8) * ks2 * ks3 * x_dim1) + 8 * ks2 * x_dim1 * ks3 * ks3)
    tmp3 = tl.load(in_ptr1 + in_ptr1_offset, x_mask, eviction_policy='evict_last')

    # Load data from in_out_ptr0
    tmp14 = tl.load(in_out_ptr0 + x_linear_index, x_mask, eviction_policy='evict_last')

    # Constants
    scale_factor = 0.16666666666666666
    bias_add = 3.0
    one = 1.0

    # Computation
    scaled_tmp0 = tmp0 * scale_factor
    biased_tmp3 = tmp3 + bias_add
    sigmoid_result = tl.sigmoid(biased_tmp3)
    product1 = scaled_tmp0 * sigmoid_result
    product2 = scaled_tmp0 * tmp3
    sigmoid_complement = one - sigmoid_result
    sigmoid_product = sigmoid_result * sigmoid_complement
    product3 = product2 * sigmoid_product
    combined_result = product1 + product3

    # Final computation
    subtracted_result = tmp14 - tmp3
    exp_result = tl.math.exp(subtracted_result)
    final_result = combined_result * exp_result

    # Store result
    tl.store(in_out_ptr0 + x_linear_index, final_result, x_mask)