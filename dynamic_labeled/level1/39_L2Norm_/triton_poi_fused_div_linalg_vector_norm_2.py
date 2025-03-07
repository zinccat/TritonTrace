# From: 39_L2Norm_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_linalg_vector_norm_2(input_ptr0, input_ptr1, output_ptr0, kernel_size, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    element_indices = block_indices
    kernel_indices = block_indices // kernel_size
    loaded_values0 = tl.load(input_ptr0 + (element_indices), valid_mask, eviction_policy='evict_last')
    loaded_values1 = tl.load(input_ptr1 + (kernel_indices), valid_mask, eviction_policy='evict_last')
    sqrt_values = tl.extra.cuda.libdevice.sqrt(loaded_values1)
    result_values = loaded_values0 / sqrt_values
    tl.store(output_ptr0 + (element_indices), result_values, valid_mask)