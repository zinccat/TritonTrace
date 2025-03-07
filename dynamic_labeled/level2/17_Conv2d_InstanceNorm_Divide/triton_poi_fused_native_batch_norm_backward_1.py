# From: 17_Conv2d_InstanceNorm_Divide

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_batch_norm_backward_1(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, kernel_size0, kernel_size1, kernel_size2, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    element_index = index
    batch_index = index // kernel_size0

    input_value = tl.load(in_ptr0 + (element_index), mask, eviction_policy='evict_last')
    grad_output = tl.load(in_out_ptr0 + (element_index), mask, eviction_policy='evict_last')
    mean = tl.load(in_ptr1 + (batch_index), mask, eviction_policy='evict_last')
    variance = tl.load(in_ptr2 + (batch_index), mask, eviction_policy='evict_last')
    gamma = tl.load(in_ptr3 + (batch_index), mask, eviction_policy='evict_last')
    beta = tl.load(in_ptr4 + (batch_index), mask, eviction_policy='evict_last')

    half = 0.5
    normalized_input = input_value * half
    centered_grad_output = grad_output - mean
    variance_factor = (
        tl.full([], 1.0, tl.float64) / ((64 * kernel_size1 + ((-64) * kernel_size1 * kernel_size2) + 16 * kernel_size1 * kernel_size2 * kernel_size2) / (16 * kernel_size1))
    )
    variance_factor_float32 = variance_factor.to(tl.float32)
    scaled_variance = variance * variance_factor_float32
    variance_squared = variance * variance
    scaled_variance_squared = scaled_variance * variance_squared
    normalized_grad_output = centered_grad_output * scaled_variance_squared
    adjusted_input = normalized_input - normalized_grad_output
    scaled_beta = beta * variance_factor_float32
    final_adjustment = adjusted_input - scaled_beta
    gamma_factor = gamma * 1.0
    final_output = final_adjustment * gamma_factor

    tl.store(in_out_ptr0 + (element_index), final_output, mask)