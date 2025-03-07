# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_162poi_fused_cat_162(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, 
    output_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xnumel = 329280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    channel_index = (xindex // 49) % 672
    spatial_index = xindex % 49
    batch_index = xindex // 32928
    linear_index = xindex

    zero_value = tl.full([1], 0, tl.int64)
    block_size_512 = tl.full([1], 512, tl.int64)
    block_size_544 = tl.full([1], 544, tl.int64)
    block_size_576 = tl.full([1], 576, tl.int64)
    block_size_608 = tl.full([1], 608, tl.int64)
    block_size_640 = tl.full([1], 640, tl.int64)
    block_size_672 = tl.full([1], 672, tl.int64)

    load_mask_512 = channel_index < block_size_512
    value_512 = tl.load(input_ptr0 + (spatial_index + 49 * channel_index + 25088 * batch_index), load_mask_512 & xmask, other=0.0)

    load_mask_544 = (channel_index >= block_size_512) & (channel_index < block_size_544)
    value_544 = tl.load(input_ptr1 + (spatial_index + 49 * ((-512) + channel_index) + 1568 * batch_index), load_mask_544 & xmask, other=0.0)

    load_mask_576 = (channel_index >= block_size_544) & (channel_index < block_size_576)
    value_576 = tl.load(input_ptr2 + (spatial_index + 49 * ((-544) + channel_index) + 1568 * batch_index), load_mask_576 & xmask, other=0.0)

    load_mask_608 = (channel_index >= block_size_576) & (channel_index < block_size_608)
    value_608 = tl.load(input_ptr3 + (spatial_index + 49 * ((-576) + channel_index) + 1568 * batch_index), load_mask_608 & xmask, other=0.0)

    load_mask_640 = (channel_index >= block_size_608) & (channel_index < block_size_640)
    value_640 = tl.load(input_ptr4 + (spatial_index + 49 * ((-608) + channel_index) + 1568 * batch_index), load_mask_640 & xmask, other=0.0)

    load_mask_672 = channel_index >= block_size_640
    value_672 = tl.load(input_ptr5 + (spatial_index + 49 * ((-640) + channel_index) + 1568 * batch_index), load_mask_672 & xmask, other=0.0)

    result_640_or_672 = tl.where(load_mask_640, value_640, value_672)
    result_608_or_above = tl.where(load_mask_608, value_608, result_640_or_672)
    result_576_or_above = tl.where(load_mask_576, value_576, result_608_or_above)
    result_544_or_above = tl.where(load_mask_544, value_544, result_576_or_above)
    result_512_or_above = tl.where(load_mask_512, value_512, result_544_or_above)

    tl.store(output_ptr0 + (linear_index), result_512_or_above, xmask)