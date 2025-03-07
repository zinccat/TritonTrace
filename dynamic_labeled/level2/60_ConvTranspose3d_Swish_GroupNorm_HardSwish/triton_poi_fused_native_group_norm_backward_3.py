# From: 60_ConvTranspose3d_Swish_GroupNorm_HardSwish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_backward_3(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, kernel_size0, kernel_size1, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    element_index = index
    channel_index = index % 4

    input0 = tl.load(in_ptr0 + (4 * element_index), mask, eviction_policy='evict_last')
    input1 = tl.load(in_ptr1 + (4 * channel_index), mask, eviction_policy='evict_last')
    input0_shifted = tl.load(in_ptr0 + (1 + 4 * element_index), mask, eviction_policy='evict_last')
    input1_shifted = tl.load(in_ptr1 + (1 + 4 * channel_index), mask, eviction_policy='evict_last')
    input0_shifted2 = tl.load(in_ptr0 + (2 + 4 * element_index), mask, eviction_policy='evict_last')
    input1_shifted2 = tl.load(in_ptr1 + (2 + 4 * channel_index), mask, eviction_policy='evict_last')
    input0_shifted3 = tl.load(in_ptr0 + (3 + 4 * element_index), mask, eviction_policy='evict_last')
    input1_shifted3 = tl.load(in_ptr1 + (3 + 4 * channel_index), mask, eviction_policy='evict_last')
    grad_output = tl.load(in_out_ptr0 + (element_index), mask)
    input2 = tl.load(in_ptr2 + (4 * element_index), mask, eviction_policy='evict_last')
    input2_shifted = tl.load(in_ptr2 + (1 + 4 * element_index), mask, eviction_policy='evict_last')
    input2_shifted2 = tl.load(in_ptr2 + (2 + 4 * element_index), mask, eviction_policy='evict_last')
    input2_shifted3 = tl.load(in_ptr2 + (3 + 4 * element_index), mask, eviction_policy='evict_last')
    input3 = tl.load(in_ptr3 + (element_index), mask)

    product0 = input0 * input1
    product1 = input0_shifted * input1_shifted
    sum0 = product0 + product1
    product2 = input0_shifted2 * input1_shifted2
    sum1 = sum0 + product2
    product3 = input0_shifted3 * input1_shifted3
    sum2 = sum1 + product3
    scaled_grad_output = sum2 * grad_output

    product4 = input2 * input1
    product5 = input2_shifted * input1_shifted
    sum3 = product4 + product5
    product6 = input2_shifted2 * input1_shifted2
    sum4 = sum3 + product6
    product7 = input2_shifted3 * input1_shifted3
    sum5 = sum4 + product7
    difference = scaled_grad_output - sum5

    scaled_difference = difference * input3
    cubed_difference = scaled_difference * scaled_difference * scaled_difference

    factor = 2.0
    kernel_size0_float = kernel_size0.to(tl.float32)
    scale_factor0 = factor * kernel_size0_float
    base = -1.0
    exponent0 = base + scale_factor0
    power0 = tl.extra.cuda.libdevice.pow(exponent0, factor)

    factor2 = 4.0
    scale_factor1 = factor2 * power0
    kernel_size1_float = kernel_size1.to(tl.float32)
    scale_factor2 = factor * kernel_size1_float
    exponent1 = base + scale_factor2
    power1 = scale_factor1 * exponent1

    divisor = tl.full([1], 1.0, tl.float64) / power1.to(tl.float64)
    normalization_factor = divisor.to(tl.float32)

    adjusted_difference = cubed_difference * normalization_factor
    neg_adjusted_difference = -adjusted_difference
    scaled_neg_adjusted_difference = neg_adjusted_difference * grad_output
    scaled_sum2 = sum2 * input3
    scaled_normalization_factor = scaled_sum2 * normalization_factor
    final_adjustment = scaled_neg_adjusted_difference - scaled_normalization_factor

    tl.store(out_ptr1 + (element_index), difference, mask)
    tl.store(in_out_ptr0 + (element_index), final_adjustment, mask)