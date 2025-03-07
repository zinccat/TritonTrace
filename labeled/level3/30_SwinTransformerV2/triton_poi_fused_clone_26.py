# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_26poi_fused_clone_26(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1505280
    program_id_offset = tl.program_id(0) * XBLOCK
    index_within_block = program_id_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = index_within_block < xnumel
    
    # Calculate indices for accessing the input tensor
    block_index = index_within_block % 1344
    channel_index = (index_within_block // 1344) % 7
    depth_index = (index_within_block // 9408) % 4
    batch_index = index_within_block // 37632
    
    # Calculate the linear index for loading and storing
    linear_index = index_within_block
    
    # Load data from the input pointer with the calculated index
    input_index = block_index + 1344 * depth_index + 5376 * channel_index + 37632 * batch_index
    loaded_data = tl.load(in_ptr0 + input_index, valid_mask)
    
    # Store the loaded data to the output pointer
    tl.store(out_ptr0 + linear_index, loaded_data, valid_mask)