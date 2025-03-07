# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_44poi_fused_cat_44(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    channel_index = (xindex // 784) % 256
    pixel_index = xindex % 784
    batch_index = xindex // 200704
    linear_index = xindex
    
    channel = channel_index
    tl.full([1], 0, tl.int64)
    
    channel_128 = tl.full([1], 128, tl.int64)
    load_condition_128 = channel < channel_128
    value_128 = tl.load(in_ptr0 + (pixel_index + 784 * channel + 100352 * batch_index), load_condition_128, other=0.0)
    
    channel_160 = tl.full([1], 160, tl.int64)
    load_condition_160 = (channel >= channel_128) & (channel < channel_160)
    value_160 = tl.load(in_ptr1 + (pixel_index + 784 * (channel - 128) + 25088 * batch_index), load_condition_160, other=0.0)
    
    channel_192 = tl.full([1], 192, tl.int64)
    load_condition_192 = (channel >= channel_160) & (channel < channel_192)
    value_192 = tl.load(in_ptr2 + (pixel_index + 784 * (channel - 160) + 25088 * batch_index), load_condition_192, other=0.0)
    
    channel_224 = tl.full([1], 224, tl.int64)
    load_condition_224 = (channel >= channel_192) & (channel < channel_224)
    value_224 = tl.load(in_ptr3 + (pixel_index + 784 * (channel - 192) + 25088 * batch_index), load_condition_224, other=0.0)
    
    channel_256 = tl.full([1], 256, tl.int64)
    load_condition_256 = channel >= channel_224
    value_256 = tl.load(in_ptr4 + (pixel_index + 784 * (channel - 224) + 25088 * batch_index), load_condition_256, other=0.0)
    
    result_224 = tl.where(load_condition_224, value_224, value_256)
    result_192 = tl.where(load_condition_192, value_192, result_224)
    result_160 = tl.where(load_condition_160, value_160, result_192)
    result_128 = tl.where(load_condition_128, value_128, result_160)
    
    final_result = result_128
    tl.store(out_ptr0 + (linear_index), final_result, None)