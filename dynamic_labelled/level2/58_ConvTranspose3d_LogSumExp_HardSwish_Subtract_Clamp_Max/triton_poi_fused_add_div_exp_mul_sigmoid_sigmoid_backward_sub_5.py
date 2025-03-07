# From: 58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_div_exp_mul_sigmoid_sigmoid_backward_sub_5poi_fused_add_div_exp_mul_sigmoid_sigmoid_backward_sub_5(
    in_out_ptr0, in_ptr0, in_ptr1, kernel_size0, kernel_size1, kernel_size2, kernel_size3, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    kernel_index0 = index % kernel_size0
    kernel_index1 = index // kernel_size1
    linear_index = index

    # Load data from in_ptr0
    load_index0 = (
        kernel_index0 + ((-1) * kernel_index1) +
        ((-4) * kernel_index1 * kernel_size3 * kernel_size3) +
        2 * kernel_size2 * kernel_index1 +
        4 * kernel_size3 * kernel_index1 +
        ((-8) * kernel_size2 * kernel_size3 * kernel_index1) +
        8 * kernel_size2 * kernel_index1 * kernel_size3 * kernel_size3
    )
    data_from_in_ptr0 = tl.load(in_ptr0 + load_index0, mask, eviction_policy='evict_last')

    # Load data from in_ptr1
    data_from_in_ptr1 = tl.load(in_ptr1 + load_index0, mask, eviction_policy='evict_last')

    # Load data from in_out_ptr0
    data_from_in_out_ptr0 = tl.load(in_out_ptr0 + linear_index, mask, eviction_policy='evict_last')

    # Constants
    scale_factor = 0.16666666666666666
    bias = 3.0
    one = 1.0

    # Computation
    scaled_data = data_from_in_ptr0 * scale_factor
    biased_data = data_from_in_ptr1 + bias
    sigmoid_output = tl.sigmoid(biased_data)
    product1 = scaled_data * sigmoid_output
    product2 = scaled_data * data_from_in_ptr1
    sigmoid_derivative = sigmoid_output * (one - sigmoid_output)
    product3 = product2 * sigmoid_derivative
    combined_result = product1 + product3

    # Final computation
    difference = data_from_in_out_ptr0 - data_from_in_ptr1
    exp_result = tl.math.exp(difference)
    final_result = combined_result * exp_result

    # Store result
    tl.store(in_out_ptr0 + linear_index, final_result, mask)