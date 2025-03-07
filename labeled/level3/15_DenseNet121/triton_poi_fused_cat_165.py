# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_165poi_fused_cat_165(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, 
    output_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xnumel = 344960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel

    # Calculate indices
    channel_index = (xindex // 49) % 704
    spatial_index = xindex % 49
    batch_index = xindex // 34496
    linear_index = xindex

    # Load and merge data from different input pointers
    base_channel = channel_index
    base_channel_512 = 512
    base_channel_544 = 544
    base_channel_576 = 576
    base_channel_608 = 608
    base_channel_640 = 640
    base_channel_672 = 672
    base_channel_704 = 704

    # Load from input_ptr0
    load_mask_0 = base_channel < base_channel_512
    data_0 = tl.load(input_ptr0 + (spatial_index + 49 * channel_index + 25088 * batch_index), load_mask_0 & xmask, other=0.0)

    # Load from input_ptr1
    load_mask_1 = (base_channel >= base_channel_512) & (base_channel < base_channel_544)
    data_1 = tl.load(input_ptr1 + (spatial_index + 49 * (channel_index - 512) + 1568 * batch_index), load_mask_1 & xmask, other=0.0)

    # Load from input_ptr2
    load_mask_2 = (base_channel >= base_channel_544) & (base_channel < base_channel_576)
    data_2 = tl.load(input_ptr2 + (spatial_index + 49 * (channel_index - 544) + 1568 * batch_index), load_mask_2 & xmask, other=0.0)

    # Load from input_ptr3
    load_mask_3 = (base_channel >= base_channel_576) & (base_channel < base_channel_608)
    data_3 = tl.load(input_ptr3 + (spatial_index + 49 * (channel_index - 576) + 1568 * batch_index), load_mask_3 & xmask, other=0.0)

    # Load from input_ptr4
    load_mask_4 = (base_channel >= base_channel_608) & (base_channel < base_channel_640)
    data_4 = tl.load(input_ptr4 + (spatial_index + 49 * (channel_index - 608) + 1568 * batch_index), load_mask_4 & xmask, other=0.0)

    # Load from input_ptr5
    load_mask_5 = (base_channel >= base_channel_640) & (base_channel < base_channel_672)
    data_5 = tl.load(input_ptr5 + (spatial_index + 49 * (channel_index - 640) + 1568 * batch_index), load_mask_5 & xmask, other=0.0)

    # Load from input_ptr6
    load_mask_6 = (base_channel >= base_channel_672) & (base_channel < base_channel_704)
    data_6 = tl.load(input_ptr6 + (spatial_index + 49 * (channel_index - 672) + 1568 * batch_index), load_mask_6 & xmask, other=0.0)

    # Merge data
    merged_data = tl.where(load_mask_5, data_5, data_6)
    merged_data = tl.where(load_mask_4, data_4, merged_data)
    merged_data = tl.where(load_mask_3, data_3, merged_data)
    merged_data = tl.where(load_mask_2, data_2, merged_data)
    merged_data = tl.where(load_mask_1, data_1, merged_data)
    merged_data = tl.where(load_mask_0, data_0, merged_data)

    # Store the result
    tl.store(output_ptr0 + (linear_index), merged_data, xmask)