# From: 24_EfficientNetB2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_29poi_fused__native_batch_norm_legit_functional_29(input_ptr_mean, input_ptr_var, input_ptr_running_mean, output_ptr_normalized, output_ptr_running_mean, output_ptr_running_var, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 192
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    indices = block_indices

    mean = tl.load(input_ptr_mean + (indices), valid_mask)
    variance = tl.load(input_ptr_mean + (192 + indices), valid_mask)
    running_mean = tl.load(input_ptr_running_mean + (indices), valid_mask)
    running_var = tl.load(input_ptr_running_var + (indices), valid_mask)

    mean_sum = mean + variance
    mean_avg = mean_sum / 2.0
    mean_diff = mean - mean_avg
    mean_sq_diff = mean_diff * mean_diff

    variance_diff = variance - mean_avg
    variance_sq_diff = variance_diff * variance_diff
    variance_sum = mean_sq_diff + variance_sq_diff
    variance_avg = variance_sum / 2.0
    variance_scaled = variance_avg * 2.0
    variance_momentum = variance_scaled * 0.1

    running_mean_scaled = running_mean * 0.9
    updated_running_mean = variance_momentum + running_mean_scaled

    running_var_scaled = running_var * 0.9
    updated_running_var = (mean_avg * 0.1) + running_var_scaled

    tl.store(output_ptr_normalized + (indices), mean_avg, valid_mask)
    tl.store(output_ptr_running_mean + (indices), updated_running_mean, valid_mask)
    tl.store(output_ptr_running_var + (indices), updated_running_var, valid_mask)