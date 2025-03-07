# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_60poi_fused__native_batch_norm_legit_functional_relu_60(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    num_elements = 3261440
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements

    input_indices = block_indices
    batch_indices = (block_indices // 784) % 416

    input_data = tl.load(input_ptr_input + (input_indices), mask)
    mean_data = tl.load(input_ptr_mean + (batch_indices), mask, eviction_policy='evict_last')
    var_data = tl.load(input_ptr_var + (batch_indices), mask, eviction_policy='evict_last')
    scale_data = tl.load(input_ptr_scale + (batch_indices), mask, eviction_policy='evict_last')
    shift_data = tl.load(input_ptr_shift + (batch_indices), mask, eviction_policy='evict_last')

    normalized_data = input_data - mean_data
    variance_scale = 7840.0
    epsilon = 1e-05

    inv_std_dev = tl.extra.cuda.libdevice.rsqrt(var_data / variance_scale + epsilon)
    scaled_data = normalized_data * inv_std_dev * scale_data
    shifted_data = scaled_data + shift_data

    relu_output = tl.full([1], 0, tl.int32)
    relu_applied = triton_helpers.maximum(relu_output, shifted_data)

    tl.store(output_ptr + (input_indices), relu_applied, mask)