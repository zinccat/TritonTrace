# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_22poi_fused_add_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    global_index = xindex
    channel_index = xindex % 192
    spatial_index = ((xindex // 192) % 784)
    batch_index = xindex // 150528

    out_value = tl.load(in_out_ptr0 + (global_index), xmask)
    input0_value = tl.load(
        in_ptr0 + (
            32 * (((4 + (spatial_index % 28)) % 7)) +
            224 * (((4 + (spatial_index // 28)) % 7)) +
            1568 * (channel_index // 32) +
            9408 * (((4 + (spatial_index % 28)) // 7)) +
            47040 * (triton_helpers.div_floor_integer(4 + (spatial_index // 28), 7)) +
            235200 * batch_index +
            ((channel_index % 32))
        ), xmask
    )
    input1_value = tl.load(
        in_ptr1 + (
            7 * (((4 + (spatial_index // 28)) % 7)) +
            49 * (channel_index // 32) +
            (((4 + (spatial_index % 28)) % 7))
        ), xmask, eviction_policy='evict_last'
    )
    input2_value = tl.load(in_ptr2 + (global_index), xmask)
    input3_value = tl.load(in_ptr3 + (channel_index), xmask, eviction_policy='evict_last')

    combined_input0_input1 = input0_value + input1_value
    combined_out_input0_input1 = out_value + combined_input0_input1
    combined_input2_input3 = input2_value + input3_value
    final_result = combined_out_input0_input1 + combined_input2_input3

    tl.store(in_out_ptr0 + (global_index), final_result, xmask)