# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_63poi_fused_clone_63(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 752640
    program_id_offset = tl.program_id(0) * XBLOCK
    index_within_block = tl.arange(0, XBLOCK)[:]
    valid_indices_mask = index_within_block < xnumel

    # Calculate indices for accessing the input tensor
    block_within_channel = index_within_block % 384
    block_within_height = (index_within_block // 384) % 7
    block_within_width = (index_within_block // 2688) % 7
    block_within_depth1 = (index_within_block // 18816) % 2
    block_within_depth2 = (index_within_block // 37632) % 2
    block_within_batch = index_within_block // 75264
    absolute_index = index_within_block

    # Calculate the offset for loading data
    load_offset = (
        block_within_channel +
        384 * (((3 + block_within_height + 7 * block_within_depth1) % 14)) +
        5376 * (((3 + block_within_width + 7 * block_within_depth2) % 14)) +
        75264 * block_within_batch
    )

    # Load data from input pointer and store it to output pointer
    tmp_data = tl.load(in_ptr0 + load_offset, valid_indices_mask)
    tl.store(out_ptr0 + absolute_index, tmp_data, valid_indices_mask)