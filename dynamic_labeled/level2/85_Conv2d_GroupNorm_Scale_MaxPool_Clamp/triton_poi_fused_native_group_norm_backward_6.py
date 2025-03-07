# From: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_backward_6(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, 
    kernel_size0, kernel_size1, kernel_size2, kernel_size3, 
    num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    x_mod_k0 = index % kernel_size0
    x_div_k0 = index // kernel_size0
    x_mod_16 = x_div_k0 % 16
    x_div_k3 = index // kernel_size3
    original_index = index

    tmp0 = tl.load(
        in_ptr0 + (((-2) * (x_mod_k0 // kernel_size1)) + 4 * x_div_k0 + 
        kernel_size2 * (x_mod_k0 // kernel_size1) + x_div_k0 * kernel_size2 * kernel_size2 + 
        ((-4) * kernel_size2 * x_div_k0)), mask, eviction_policy='evict_last'
    )
    tmp1 = tl.load(in_ptr1 + (x_mod_16), mask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x_div_k3), mask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x_mod_16), mask, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (original_index), mask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x_div_k3), mask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x_div_k3), mask, eviction_policy='evict_last')

    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 * tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp12 = tmp10 + tmp11

    tl.store(in_out_ptr0 + (original_index), tmp12, mask)