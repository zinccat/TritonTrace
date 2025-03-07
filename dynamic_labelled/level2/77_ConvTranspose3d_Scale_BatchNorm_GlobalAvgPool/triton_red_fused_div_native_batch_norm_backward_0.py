# From: 77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_div_native_batch_norm_backward_0(
    input_ptr, output_ptr, kernel_size, input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 32
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_indices = input_index
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_divisor = reduction_index // kernel_size
        loaded_values = tl.load(
            input_ptr + (input_indices + 32 * reduction_divisor), 
            reduction_mask & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        kernel_size_float = kernel_size.to(tl.float32)
        normalized_values = loaded_values / kernel_size_float
        broadcasted_values = tl.broadcast_to(normalized_values, [XBLOCK, RBLOCK])
        temp_sum += broadcasted_values
        temp_sum = tl.where(reduction_mask & input_mask, temp_sum, temp_sum)
    
    summed_values = tl.sum(temp_sum, 1)[:, None]
    tl.store(output_ptr + (input_indices), summed_values, input_mask)