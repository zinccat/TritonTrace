# From: 91_cumsum_reverse

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_flip_1(in_ptr0, out_ptr0, kernel_size, num_elements, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    element_index = block_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = element_index < num_elements
    kernel_index = element_index % kernel_size
    batch_index = element_index // kernel_size
    global_index = element_index
    
    # Calculate the memory address for loading
    load_address = (-1) + kernel_size + ((-1) * kernel_index) + kernel_size * batch_index
    temp_value = tl.load(in_ptr0 + load_address, valid_mask, eviction_policy='evict_last')
    
    # Store the result in the output pointer
    tl.store(out_ptr0 + global_index, temp_value, valid_mask)