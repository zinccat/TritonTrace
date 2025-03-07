# From: 24_EfficientNetB2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_17poi_fused__native_batch_norm_legit_functional_17(input_ptr_mean, input_ptr_var, input_ptr_running_mean, output_ptr_normalized, output_ptr_running_mean, output_ptr_running_var, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 576
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    indices = block_indices

    mean = tl.load(input_ptr_mean + (indices), valid_mask)
    var = tl.load(input_ptr_mean + (576 + indices), valid_mask)
    running_mean = tl.load(input_ptr_running_mean + (indices), valid_mask)
    running_var = tl.load(input_ptr_running_var + (indices), valid_mask)

    mean_sum = mean + var
    mean_avg = mean_sum / 2.0
    mean_diff = mean - mean_avg
    mean_sq_diff = mean_diff * mean_diff

    var_diff = var - mean_avg
    var_sq_diff = var_diff * var_diff
    var_sum_sq_diff = mean_sq_diff + var_sq_diff
    var_avg_sq_diff = var_sum_sq_diff / 2.0
    var_unbiased = var_avg_sq_diff * 2.0
    var_momentum = var_unbiased * 0.1

    running_mean_momentum = running_mean * 0.9
    updated_running_mean = var_momentum + running_mean_momentum

    running_var_momentum = running_var * 0.9
    updated_running_var = (mean_avg * 0.1) + running_var_momentum

    tl.store(output_ptr_normalized + (indices), mean_avg, valid_mask)
    tl.store(output_ptr_running_mean + (indices), updated_running_mean, valid_mask)
    tl.store(output_ptr_running_var + (indices), updated_running_var, valid_mask)