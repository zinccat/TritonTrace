# From: 43_Conv3d_Max_LogSumExp_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_logsumexp_mul_threshold_backward_0poi_fused_logsumexp_mul_threshold_backward_0(
    in_out_ptr0, in_ptr0, in_ptr1, kernel_size0, kernel_size1, kernel_size2, kernel_size3, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    kernel_index0 = index % kernel_size0
    kernel_index2 = index // kernel_size1
    linear_index = index
    loaded_value0 = tl.load(
        in_ptr0 + (kernel_index0 + kernel_index2 * (kernel_size3 // 2) * (kernel_size3 // 2) * (kernel_size2 // 2)),
        mask,
        eviction_policy='evict_last'
    ).to(tl.int1)
    loaded_value1 = tl.load(
        in_ptr1 + (kernel_index0 + kernel_index2 * (kernel_size3 // 2) * (kernel_size3 // 2) * (kernel_size2 // 2)),
        mask,
        eviction_policy='evict_last'
    )
    loaded_value4 = tl.load(in_out_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    zero_value = 0.0
    conditional_value = tl.where(loaded_value0, zero_value, loaded_value1)
    result_value = conditional_value * loaded_value4
    tl.store(in_out_ptr0 + (linear_index), result_value, mask)