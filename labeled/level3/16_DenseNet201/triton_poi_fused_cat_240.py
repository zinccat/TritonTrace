# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_240poi_fused_cat_240(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, 
    output_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xnumel = 517440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel

    # Calculate indices
    block_index = (xindex // 49) % 1056
    within_block_index = xindex % 49
    layer_index = xindex // 51744
    flat_index = xindex

    # Temporary variables for conditions
    tmp_block_index = block_index
    zero_value = tl.full([1], 0, tl.int64)
    threshold_896 = tl.full([1], 896, tl.int64)
    threshold_928 = tl.full([1], 928, tl.int64)
    threshold_960 = tl.full([1], 960, tl.int64)
    threshold_992 = tl.full([1], 992, tl.int64)
    threshold_1024 = tl.full([1], 1024, tl.int64)
    threshold_1056 = tl.full([1], 1056, tl.int64)

    # Load and conditionally select values
    load_0 = tl.load(input_ptr0 + (within_block_index + 49 * block_index + 43904 * layer_index), 
                     (tmp_block_index < threshold_896) & xmask, other=0.0)
    load_1 = tl.load(input_ptr1 + (within_block_index + 49 * (block_index - 896) + 1568 * layer_index), 
                     ((tmp_block_index >= threshold_896) & (tmp_block_index < threshold_928)) & xmask, other=0.0)
    load_2 = tl.load(input_ptr2 + (within_block_index + 49 * (block_index - 928) + 1568 * layer_index), 
                     ((tmp_block_index >= threshold_928) & (tmp_block_index < threshold_960)) & xmask, other=0.0)
    load_3 = tl.load(input_ptr3 + (within_block_index + 49 * (block_index - 960) + 1568 * layer_index), 
                     ((tmp_block_index >= threshold_960) & (tmp_block_index < threshold_992)) & xmask, other=0.0)
    load_4 = tl.load(input_ptr4 + (within_block_index + 49 * (block_index - 992) + 1568 * layer_index), 
                     ((tmp_block_index >= threshold_992) & (tmp_block_index < threshold_1024)) & xmask, other=0.0)
    load_5 = tl.load(input_ptr5 + (within_block_index + 49 * (block_index - 1024) + 1568 * layer_index), 
                     (tmp_block_index >= threshold_1024) & xmask, other=0.0)

    # Conditional selection
    selected_4 = tl.where((tmp_block_index >= threshold_992) & (tmp_block_index < threshold_1024), load_4, load_5)
    selected_3 = tl.where((tmp_block_index >= threshold_960) & (tmp_block_index < threshold_992), load_3, selected_4)
    selected_2 = tl.where((tmp_block_index >= threshold_928) & (tmp_block_index < threshold_960), load_2, selected_3)
    selected_1 = tl.where((tmp_block_index >= threshold_896) & (tmp_block_index < threshold_928), load_1, selected_2)
    selected_0 = tl.where(tmp_block_index < threshold_896, load_0, selected_1)

    # Store the result
    tl.store(output_ptr0 + (flat_index), selected_0, xmask)