# From: 34_VanillaRNNHidden

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_tanh_1poi_fused_addmm_tanh_1(output_ptr, input_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 2048
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    global_indices = block_indices
    local_indices = block_indices % 256
    output_values = tl.load(output_ptr + (global_indices), valid_mask)
    input_values = tl.load(input_ptr + (local_indices), valid_mask, eviction_policy='evict_last')
    sum_values = output_values + input_values
    tanh_values = tl.extra.cuda.libdevice.tanh(sum_values)
    tl.store(output_ptr + (global_indices), tanh_values, valid_mask)