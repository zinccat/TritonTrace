# From: 43_Conv3d_Max_LogSumExp_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_logsumexp_mul_threshold_backward_0(
    in_out_ptr0, in_ptr0, in_ptr1, kernel_size_0, kernel_size_1, kernel_size_2, kernel_size_3, 
    num_elements, BLOCK_SIZE : tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements

    kernel_index_0 = block_indices % kernel_size_0
    kernel_index_2 = block_indices // kernel_size_1
    linear_index = block_indices

    input_masked = tl.load(
        in_ptr0 + (kernel_index_0 + kernel_index_2 * (kernel_size_3 // 2) * (kernel_size_3 // 2) * (kernel_size_2 // 2)), 
        valid_mask, 
        eviction_policy='evict_last'
    ).to(tl.int1)

    input_data_1 = tl.load(
        in_ptr1 + (kernel_index_0 + kernel_index_2 * (kernel_size_3 // 2) * (kernel_size_3 // 2) * (kernel_size_2 // 2)), 
        valid_mask, 
        eviction_policy='evict_last'
    )

    input_output_data = tl.load(in_out_ptr0 + (linear_index), valid_mask, eviction_policy='evict_last')

    zero_value = 0.0
    selected_input = tl.where(input_masked, zero_value, input_data_1)

    result = selected_input * input_output_data

    tl.store(in_out_ptr0 + (linear_index), result, valid_mask)