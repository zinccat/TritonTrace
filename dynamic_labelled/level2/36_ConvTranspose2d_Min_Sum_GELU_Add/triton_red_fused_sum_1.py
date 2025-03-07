# From: 36_ConvTranspose2d_Min_Sum_GELU_Add

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_sum_1(in_ptr0, out_ptr0, kernel_size0, kernel_size1, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_num_elements = 16
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_indices = input_index
    _accumulated_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        kernel_index0 = reduction_index % kernel_size0
        kernel_index1 = reduction_index // kernel_size0
        
        loaded_values = tl.load(
            in_ptr0 + (kernel_index0 + 2 * kernel_size1 * input_indices + 32 * kernel_size1 * kernel_index1),
            reduction_mask & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        updated_sum = _accumulated_sum + broadcasted_values
        _accumulated_sum = tl.where(reduction_mask & input_mask, updated_sum, _accumulated_sum)
    
    summed_values = tl.sum(_accumulated_sum, 1)[:, None]
    tl.store(out_ptr0 + (input_indices), summed_values, input_mask)