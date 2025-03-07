# From: 61_ConvTranspose3d_ReLU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_backward_4poi_fused_native_group_norm_backward_4(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, 
    output_ptr0, kernel_size0, kernel_size1, kernel_size2, kernel_size3, num_elements, 
    XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    div_kernel0 = index // kernel_size0
    mod_kernel1 = (index // kernel_size1) % 128
    mod_kernel0 = index % kernel_size1
    div_kernel1 = index // kernel_size1

    input0 = tl.load(input_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    input1 = tl.load(input_ptr1 + (div_kernel0), mask, eviction_policy='evict_last')
    input2 = tl.load(input_ptr2 + (mod_kernel1), mask, eviction_policy='evict_last')

    complex_index = (
        2 * (((mod_kernel0 // (2 + kernel_size2)) % (2 + kernel_size2))) +
        4 * (mod_kernel0 // (4 + kernel_size2 * kernel_size2 + 4 * kernel_size2)) +
        8 * div_kernel1 +
        kernel_size2 * (((mod_kernel0 // (2 + kernel_size2)) % (2 + kernel_size2))) +
        kernel_size2 * kernel_size2 * (mod_kernel0 // (4 + kernel_size2 * kernel_size2 + 4 * kernel_size2)) +
        2 * div_kernel1 * kernel_size2 * kernel_size2 +
        4 * kernel_size2 * (mod_kernel0 // (4 + kernel_size2 * kernel_size2 + 4 * kernel_size2)) +
        4 * kernel_size3 * div_kernel1 +
        8 * kernel_size2 * div_kernel1 +
        kernel_size3 * div_kernel1 * kernel_size2 * kernel_size2 +
        4 * kernel_size2 * kernel_size3 * div_kernel1 +
        (mod_kernel0 % (2 + kernel_size2))
    )

    input3 = tl.load(input_ptr3 + (complex_index), mask, eviction_policy='evict_last')
    input4 = tl.load(input_ptr4 + (div_kernel0), mask, eviction_policy='evict_last')
    input5 = tl.load(input_ptr5 + (div_kernel0), mask, eviction_policy='evict_last')
    input6 = tl.load(input_ptr6 + (div_kernel0), mask, eviction_policy='evict_last')

    product1 = input1 * input2
    product2 = input0 * product1
    zero_tensor = tl.full([1], 0, tl.int32)
    max_value = triton_helpers.maximum(zero_tensor, input3)
    product3 = input4 * input5
    difference = product3 - input6
    product4 = difference * input1
    product5 = product4 * input1
    product6 = product5 * input1
    constant16 = 2.0
    kernel_size2_float = kernel_size2.to(tl.float32)
    sum16_kernel2 = constant16 + kernel_size2_float
    power_result = tl.extra.cuda.libdevice.pow(sum16_kernel2, constant16)
    constant16_2 = 16.0
    product7 = constant16_2 * power_result
    kernel_size3_float = kernel_size3.to(tl.float32)
    sum16_kernel3 = constant16 + kernel_size3_float
    product8 = product7 * sum16_kernel3
    product8_double = product8.to(tl.float64)
    one_tensor = tl.full([1], 1.0, tl.float64)
    division_result = one_tensor / product8_double
    division_result_float = division_result.to(tl.float32)
    final_product = product6 * division_result_float
    result = max_value * final_product
    output = product2 + result

    tl.store(output_ptr0 + (linear_index), output, mask)