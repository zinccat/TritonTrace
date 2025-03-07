# From: 65_Conv2d_AvgPool_Sigmoid_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_sigmoid_sum_2(in_ptr0, out_ptr0, kernel_size, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_indices_0 = input_indices
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_indices = reduction_offset + reduction_base
        reduction_mask = reduction_indices < reduction_num_elements
        reduction_indices_1 = reduction_indices
        temp_load = tl.load(
            in_ptr0 + (reduction_indices_1 + 16 * input_indices_0 + ((-32) * input_indices_0 * (kernel_size // 2)) + 16 * input_indices_0 * (kernel_size // 2) * (kernel_size // 2)),
            reduction_mask & input_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        temp_sigmoid = tl.sigmoid(temp_load)
        temp_broadcast = tl.broadcast_to(temp_sigmoid, [XBLOCK, RBLOCK])
        temp_accumulate = temp_sum + temp_broadcast
        temp_sum = tl.where(reduction_mask & input_mask, temp_accumulate, temp_sum)
    
    temp_final_sum = tl.sum(temp_sum, 1)[:, None]
    tl.store(out_ptr0 + (input_indices_0), temp_final_sum, input_mask)