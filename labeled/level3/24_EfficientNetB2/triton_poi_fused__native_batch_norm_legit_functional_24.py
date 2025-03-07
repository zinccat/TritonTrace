# From: 24_EfficientNetB2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_24poi_fused__native_batch_norm_legit_functional_24(input_ptr_mean, input_ptr_var, input_ptr_running_mean, output_ptr_normalized, output_ptr_running_mean, output_ptr_running_var, num_elements, XBLOCK: tl.constexpr):
    num_elements = 864
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    base_indices = indices

    mean = tl.load(input_ptr_mean + (base_indices), mask)
    variance = tl.load(input_ptr_mean + (864 + base_indices), mask)
    running_mean = tl.load(input_ptr_running_mean + (base_indices), mask)
    running_var = tl.load(input_ptr_running_var + (base_indices), mask)

    mean_sum = mean + variance
    mean_divisor = 2.0
    mean_centered = mean - (mean_sum / mean_divisor)
    mean_centered_sq = mean_centered * mean_centered

    variance_centered = variance - (mean_sum / mean_divisor)
    variance_centered_sq = variance_centered * variance_centered

    variance_sum = mean_centered_sq + variance_centered_sq
    variance_avg = variance_sum / mean_divisor
    variance_scaled = variance_avg * mean_divisor

    momentum = 0.1
    variance_momentum = variance_scaled * momentum

    running_mean_momentum = running_mean * 0.9
    updated_running_mean = variance_momentum + running_mean_momentum

    mean_momentum = (mean_sum / mean_divisor) * momentum
    running_var_momentum = running_var * 0.9
    updated_running_var = mean_momentum + running_var_momentum

    tl.store(output_ptr_normalized + (base_indices), mean_sum / mean_divisor, mask)
    tl.store(output_ptr_running_mean + (base_indices), updated_running_mean, mask)
    tl.store(output_ptr_running_var + (base_indices), updated_running_var, mask)