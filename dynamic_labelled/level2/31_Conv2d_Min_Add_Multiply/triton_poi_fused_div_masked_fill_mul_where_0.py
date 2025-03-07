# From: 31_Conv2d_Min_Add_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_masked_fill_mul_where_0poi_fused_div_masked_fill_mul_where_0(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    indices = block_indices

    input_data0 = tl.load(input_ptr0 + (indices), valid_mask).to(tl.int1)
    input_data1 = tl.load(input_ptr1 + (indices), valid_mask).to(tl.int1)
    input_data2 = tl.load(input_ptr2 + (indices), valid_mask)

    multiplier = 2.0
    multiplied_data = input_data2 * multiplier

    divisor = 0.5
    divided_data = multiplied_data * divisor

    conditional_data = tl.where(input_data1, divided_data, multiplied_data)

    fill_value = 0.0
    final_data = tl.where(input_data0, fill_value, conditional_data)

    tl.store(output_ptr0 + (indices), final_data, valid_mask)