# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_9poi_fused_add_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    # Descriptive variable names
    block_index = xindex
    sub_block_index = xindex % 96
    mid_block_index = (xindex // 96) % 3136
    outer_block_index = xindex // 301056
    
    # Load operations
    output_value = tl.load(in_out_ptr0 + (block_index), None)
    input_value0 = tl.load(
        in_ptr0 + (
            32 * (((4 + (mid_block_index % 56)) % 7)) +
            224 * (((4 + (mid_block_index // 56)) % 7)) +
            1568 * (sub_block_index // 32) +
            4704 * (((4 + (mid_block_index % 56)) // 7)) +
            42336 * (triton_helpers.div_floor_integer(4 + (mid_block_index // 56), 7)) +
            381024 * outer_block_index +
            ((sub_block_index % 32))
        ), None
    )
    input_value1 = tl.load(
        in_ptr1 + (
            7 * (((4 + (mid_block_index // 56)) % 7)) +
            49 * (sub_block_index // 32) +
            (((4 + (mid_block_index % 56)) % 7))
        ), None, eviction_policy='evict_last'
    )
    input_value2 = tl.load(in_ptr2 + (block_index), None)
    input_value3 = tl.load(in_ptr3 + (sub_block_index), None, eviction_policy='evict_last')
    
    # Computation
    intermediate_sum1 = input_value0 + input_value1
    intermediate_sum2 = output_value + intermediate_sum1
    intermediate_sum3 = input_value2 + input_value3
    final_result = intermediate_sum2 + intermediate_sum3
    
    # Store result
    tl.store(in_out_ptr0 + (block_index), final_result, None)