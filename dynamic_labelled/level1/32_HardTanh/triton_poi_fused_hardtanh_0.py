# From: 32_HardTanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_hardtanh_0poi_fused_hardtanh_0(in_ptr0, out_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    indices = block_indices
    input_values = tl.load(in_ptr0 + (indices), valid_mask)
    lower_bound = -1.0
    clamped_values = triton_helpers.maximum(input_values, lower_bound)
    upper_bound = 1.0
    hardtanh_values = triton_helpers.minimum(clamped_values, upper_bound)
    tl.store(out_ptr0 + (indices), hardtanh_values, valid_mask)