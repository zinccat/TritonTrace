# From: 41_Gemm_BatchNorm_GELU_GroupNorm_Mean_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_batch_norm_backward_4(
    in_out_ptr, input_data, mean_ptr, variance_ptr, scale_ptr, shift_ptr, running_var_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < num_elements
    x2 = xindex
    x0 = (xindex % 1024)
    
    output_grad = tl.load(in_out_ptr + (x2), xmask)
    input_grad = tl.load(input_data + (x2), xmask)
    batch_mean = tl.load(mean_ptr + (x0), xmask, eviction_policy='evict_last')
    batch_variance = tl.load(variance_ptr + (x0), xmask, eviction_policy='evict_last')
    scale = tl.load(scale_ptr + (x0), xmask, eviction_policy='evict_last')
    shift = tl.load(shift_ptr + (x0), xmask, eviction_policy='evict_last')
    running_variance = tl.load(running_var_ptr + (x0), xmask, eviction_policy='evict_last')
    
    centered_input = input_grad - batch_mean
    inv_std_dev = (tl.full([], 1.00000000000000, tl.float64) / ((1024 * kernel_size) / 1024)).to(tl.float32)
    std_dev = inv_std_dev * batch_variance
    variance_term = std_dev * batch_variance
    scaled_centered_input = centered_input * variance_term
    grad_input = output_grad - scaled_centered_input
    scaled_shift = shift * inv_std_dev
    grad_output = grad_input - scaled_shift
    running_std_dev = batch_variance * running_variance
    final_grad = grad_output * running_std_dev
    
    tl.store(in_out_ptr + (x2), final_grad, xmask)