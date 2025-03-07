# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_40poi_fused_clone_40(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1505280
    program_id_offset = tl.program_id(0) * XBLOCK
    index_within_block = tl.arange(0, XBLOCK)[:]
    valid_indices_mask = index_within_block < xnumel

    # Calculate indices for accessing the input tensor
    block_offset = index_within_block % 192
    depth_index_1 = (index_within_block // 192) % 7
    depth_index_2 = (index_within_block // 1344) % 7
    height_index_1 = (index_within_block // 9408) % 4
    height_index_2 = (index_within_block // 37632) % 4
    batch_index = index_within_block // 150528
    linear_index = index_within_block

    # Calculate the memory offset for loading data
    memory_offset = (
        block_offset +
        192 * (((3 + depth_index_1 + 7 * height_index_1) % 28)) +
        5376 * (((3 + depth_index_2 + 7 * height_index_2) % 28)) +
        150528 * batch_index
    )

    # Load data from input pointer and store it to output pointer
    loaded_data = tl.load(in_ptr0 + memory_offset, valid_indices_mask)
    tl.store(out_ptr0 + (linear_index), loaded_data, valid_indices_mask)