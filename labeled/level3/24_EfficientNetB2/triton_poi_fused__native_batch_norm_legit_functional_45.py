# From: 24_EfficientNetB2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_45poi_fused__native_batch_norm_legit_functional_45(input_ptr_mean, input_ptr_var, input_ptr_running_mean, output_ptr_normalized, output_ptr_running_mean, output_ptr_running_var, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    # Load mean and variance
    mean = tl.load(input_ptr_mean + (x0), xmask)
    variance = tl.load(input_ptr_mean + (1408 + x0), xmask)

    # Load running mean and variance
    running_mean = tl.load(input_ptr_running_mean + (x0), xmask)
    running_var = tl.load(input_ptr_running_var + (x0), xmask)

    # Calculate batch mean
    batch_mean = (mean + variance) / 2.0

    # Calculate batch variance
    mean_diff = mean - batch_mean
    variance_diff = variance - batch_mean
    mean_diff_squared = mean_diff * mean_diff
    variance_diff_squared = variance_diff * variance_diff
    batch_variance = (mean_diff_squared + variance_diff_squared) / 2.0

    # Normalize variance
    normalized_variance = batch_variance * 2.0

    # Update running mean and variance
    momentum = 0.9
    epsilon = 0.1
    updated_running_mean = running_mean * momentum + normalized_variance * epsilon
    updated_running_var = running_var * momentum + batch_mean * epsilon

    # Store results
    tl.store(output_ptr_normalized + (x0), batch_mean, xmask)
    tl.store(output_ptr_running_mean + (x0), updated_running_mean, xmask)
    tl.store(output_ptr_running_var + (x0), updated_running_var, xmask)