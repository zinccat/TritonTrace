# From: 23_Conv3d_GroupNorm_Mean

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_backward_3(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, kernel_size0, kernel_size1, kernel_size2, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    batch_index = index // kernel_size0
    channel_index = index // kernel_size1
    group_index = (index // kernel_size2) % 16
    flat_index = index

    input0 = tl.load(in_ptr0 + (batch_index), mask, eviction_policy='evict_last')
    input1 = tl.load(in_ptr1 + (channel_index), mask, eviction_policy='evict_last')
    input2 = tl.load(in_ptr2 + (group_index), mask, eviction_policy='evict_last')
    grad_output = tl.load(in_out_ptr0 + (flat_index), mask, eviction_policy='evict_last')
    input3 = tl.load(in_ptr3 + (channel_index), mask, eviction_policy='evict_last')
    input4 = tl.load(in_ptr4 + (channel_index), mask, eviction_policy='evict_last')

    kernel_size0_float = kernel_size0.to(tl.float32)
    normalized_input0 = input0 / kernel_size0_float
    scaled_input = normalized_input0 * (input1 * input2)
    scaled_grad_output = grad_output * input3
    sum_scaled = scaled_input + scaled_grad_output
    final_result = sum_scaled + input4

    tl.store(in_out_ptr0 + (flat_index), final_result, mask)