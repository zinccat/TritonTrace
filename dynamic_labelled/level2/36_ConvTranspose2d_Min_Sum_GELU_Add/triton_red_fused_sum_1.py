# From: 36_ConvTranspose2d_Min_Sum_GELU_Add

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_sum_1red_fused_sum_1(input_ptr, output_ptr, kernel_size_0, kernel_size_1, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_num_elements = 16
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_0 = input_index
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_dim_1 = (reduction_index % kernel_size_0)
        reduction_dim_2 = reduction_index // kernel_size_0
        temp_load = tl.load(input_ptr + (reduction_dim_1 + 2 * kernel_size_1 * input_0 + 32 * kernel_size_1 * reduction_dim_2), reduction_mask & input_mask, eviction_policy='evict_last', other=0.0)
        temp_broadcast = tl.broadcast_to(temp_load, [XBLOCK, RBLOCK])
        temp_sum = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(reduction_mask & input_mask, temp_sum, temp_accumulator)
    
    temp_result = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (input_0), temp_result, input_mask)