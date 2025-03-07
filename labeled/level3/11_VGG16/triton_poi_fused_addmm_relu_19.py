# From: 11_VGG16

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_relu_19poi_fused_addmm_relu_19(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    global_indices = block_indices
    local_indices = block_indices % 4096
    output_values = tl.load(in_out_ptr0 + (global_indices), None)
    input_values = tl.load(in_ptr0 + (local_indices), None, eviction_policy='evict_last')
    summed_values = output_values + input_values
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_values = triton_helpers.maximum(zero_tensor, summed_values)
    tl.store(in_out_ptr0 + (global_indices), relu_values, None)