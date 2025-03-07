# From: 24_EfficientNetB2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_22poi_fused__native_batch_norm_legit_functional_22(input_ptr_mean, input_ptr_var, input_ptr_running_mean, output_ptr_normalized, output_ptr_running_mean, output_ptr_running_var, xnumel, XBLOCK: tl.constexpr):
    xnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    # Load mean and variance
    mean = tl.load(input_ptr_mean + (x0), xmask)
    variance = tl.load(input_ptr_mean + (144 + x0), xmask)

    # Load running mean and variance
    running_mean = tl.load(input_ptr_running_mean + (x0), xmask)
    running_var = tl.load(input_ptr_running_var + (x0), xmask)

    # Calculate mean and variance
    mean_sum = mean + variance
    mean_divisor = 2.0
    mean_centered = mean - (mean_sum / mean_divisor)
    mean_centered_sq = mean_centered * mean_centered
    variance_centered = variance - (mean_sum / mean_divisor)
    variance_centered_sq = variance_centered * variance_centered
    variance_sum = mean_centered_sq + variance_centered_sq
    variance_avg = variance_sum / mean_divisor
    variance_scaled = variance_avg * mean_divisor

    # Update running mean and variance
    momentum = 0.1
    updated_running_mean = running_mean * 0.9 + variance_scaled * momentum
    updated_running_var = running_var * 0.9 + (mean * momentum)

    # Store results
    tl.store(output_ptr_normalized + (x0), mean, xmask)
    tl.store(output_ptr_running_mean + (x0), updated_running_mean, xmask)
    tl.store(output_ptr_running_var + (x0), updated_running_var, xmask)