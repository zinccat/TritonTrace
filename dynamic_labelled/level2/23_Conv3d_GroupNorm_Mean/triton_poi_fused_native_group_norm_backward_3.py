# From: 23_Conv3d_GroupNorm_Mean

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_backward_3(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, kernel_size0, kernel_size1, kernel_size2, xnumel, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < xnumel
    x_depth = x_index // kernel_size0
    x_height = x_index // kernel_size1
    x_width = (x_index // kernel_size2) % 16
    x_linear_index = x_index

    input_feature_map = tl.load(in_ptr0 + (x_depth), x_mask, eviction_policy='evict_last')
    input_group_norm = tl.load(in_ptr1 + (x_height), x_mask, eviction_policy='evict_last')
    input_group_norm_weight = tl.load(in_ptr2 + (x_width), x_mask, eviction_policy='evict_last')
    input_output = tl.load(in_out_ptr0 + (x_linear_index), x_mask, eviction_policy='evict_last')
    input_group_norm_bias = tl.load(in_ptr3 + (x_height), x_mask, eviction_policy='evict_last')
    input_group_norm_running_var = tl.load(in_ptr4 + (x_height), x_mask, eviction_policy='evict_last')

    kernel_size0_float = kernel_size0.to(tl.float32)
    normalized_input_feature_map = input_feature_map / kernel_size0_float
    group_norm_weighted_input = normalized_input_feature_map * input_group_norm_weight
    group_norm_bias_weighted_input = input_output * input_group_norm_bias
    group_norm_output = group_norm_weighted_input + group_norm_bias_weighted_input
    final_output = group_norm_output + input_group_norm_running_var

    tl.store(in_out_ptr0 + (x_linear_index), final_output, x_mask)