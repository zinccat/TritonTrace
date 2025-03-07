# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_13poi_fused_cat_13(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 752640
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = block_indices < xnumel

    element_index = block_indices % 96
    row_index = (block_indices // 96) % 28
    channel_index = block_indices // 2688
    linear_index = block_indices // 96

    input_offset = 5472 + element_index + 192 * row_index + 10752 * channel_index
    output_offset = element_index + 384 * linear_index

    temp_data = tl.load(in_ptr0 + input_offset, valid_mask)
    tl.store(out_ptr0 + output_offset, temp_data, valid_mask)