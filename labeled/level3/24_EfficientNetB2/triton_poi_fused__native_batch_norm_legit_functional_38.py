# From: 24_EfficientNetB2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_38poi_fused__native_batch_norm_legit_functional_38(input_ptr_mean, input_ptr_var, input_ptr_running_mean, output_ptr_normalized, output_ptr_running_mean, output_ptr_running_var, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    # Load mean and variance
    mean = tl.load(input_ptr_mean + (x0), xmask)
    variance = tl.load(input_ptr_mean + (1728 + x0), xmask)

    # Load running mean and variance
    running_mean = tl.load(input_ptr_running_mean + (x0), xmask)
    running_var = tl.load(input_ptr_running_var + (x0), xmask)

    # Calculate mean and variance
    mean_sum = mean + variance
    mean_avg = mean_sum / 2.0
    mean_diff = mean - mean_avg
    variance_diff = variance - mean_avg

    mean_squared = mean_diff * mean_diff
    variance_squared = variance_diff * variance_squared

    variance_sum = mean_squared + variance_squared
    variance_avg = variance_sum / 2.0
    variance_scaled = variance_avg * 2.0

    # Update running mean and variance
    momentum = 0.1
    running_mean_momentum = running_mean * 0.9
    updated_running_mean = variance_scaled * momentum + running_mean_momentum

    running_var_momentum = running_var * 0.9
    updated_running_var = mean_avg * momentum + running_var_momentum

    # Store results
    tl.store(output_ptr_normalized + (x0), mean_avg, xmask)
    tl.store(output_ptr_running_mean + (x0), updated_running_mean, xmask)
    tl.store(output_ptr_running_var + (x0), updated_running_var, xmask)