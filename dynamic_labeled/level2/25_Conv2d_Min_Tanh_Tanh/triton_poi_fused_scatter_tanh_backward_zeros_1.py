# From: 25_Conv2d_Min_Tanh_Tanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_scatter_tanh_backward_zeros_1(
    input_grad_ptr, input_data_ptr, input_tanh_ptr, input_mask_ptr, output_grad_ptr, kernel_size_0, kernel_size_1, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    kernel_index_0 = index % kernel_size_0
    kernel_index_1 = index // kernel_size_0

    grad_input = tl.load(input_grad_ptr + (linear_index), mask, eviction_policy='evict_last')
    data_input = tl.load(input_data_ptr + (linear_index), mask, eviction_policy='evict_last')
    tanh_input = tl.load(input_tanh_ptr + (linear_index), mask, eviction_policy='evict_last')
    mask_input = tl.load(input_mask_ptr + (linear_index), mask, eviction_policy='evict_last')

    tl.device_assert(((0 <= grad_input) & (grad_input < 16)) | ~mask, "index out of bounds: 0 <= grad_input < 16")

    tanh_squared = tanh_input * tanh_input
    one_minus_tanh_squared = 1.0 - tanh_squared
    grad_tanh = data_input * one_minus_tanh_squared
    grad_output = grad_tanh * mask_input

    output_index = (
        kernel_index_0 + 4 * grad_input + 64 * kernel_index_1 +
        grad_input * kernel_size_1 * kernel_size_1 +
        (-64) * kernel_size_1 * kernel_index_1 +
        (-4) * kernel_size_1 * grad_input +
        16 * kernel_index_1 * kernel_size_1 * kernel_size_1
    )

    tl.store(output_grad_ptr + (output_index), grad_output, mask)