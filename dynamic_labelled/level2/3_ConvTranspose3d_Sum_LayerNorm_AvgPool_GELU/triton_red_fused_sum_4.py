# From: 3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_sum_4(input_ptr0, output_ptr0, kernel_size0, kernel_size1, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    reduction_mask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_indices = input_index
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_valid_mask = reduction_index < reduction_num_elements
        reduction_indices = reduction_index
        loaded_values = tl.load(
            input_ptr0 + (64 * (((reduction_indices + 64 * kernel_size0 * kernel_size1 * input_indices) // 64) % (8192 * kernel_size0 * kernel_size1))) + ((reduction_indices % 64)),
            reduction_valid_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        temp_sum = temp_accumulator + broadcasted_values
        temp_accumulator = tl.where(reduction_valid_mask, temp_sum, temp_accumulator)
    
    reduced_sum = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr0 + (input_indices), reduced_sum, None)